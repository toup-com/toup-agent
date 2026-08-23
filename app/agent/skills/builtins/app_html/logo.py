"""Every app gets a mark that says what it is — and looks designed.

Round 20, then corrected. The first version produced exactly what it asked
for and that turned out to be the problem: a flat pictogram centred on a
coloured rounded square, in a colour picked from a twelve-hue ramp indexed by
a hash of the slug. Generated, looked at, and the verdict was fair — twenty
apps came out as twenty interchangeable badges that belonged to no app in
particular.

Three defects, and each has its own fix here, each verified by generating a
set and looking at it rather than by reasoning about prompts:

1.  **The subject was the category's stock glyph.** A clock for a timer, a
    document for a budget, wavy lines for tides. Banning them in the drawing
    prompt did nothing — asked to choose and draw in one breath, a model
    reaches for the nearest UI pictogram no matter what the prompt says. So
    the job is SPLIT: :func:`choose_subject` names the subject in words with
    a cheap call, and :func:`draw_mark` draws the subject it is given. Naming
    is a different task with a different failure mode, and — the part that
    matters — its answer is a sentence, so a stock subject can be rejected
    before a single path is drawn. Measured: the same model that produced a
    clock, a document and a spreadsheet in one step produced "tomato with
    time wedge", "coins with one lifted" and "guitar strings and fingertips"
    in two.

2.  **The colour came from a hash.** It now comes from
    :mod:`palette`, which reads the app's own CSS custom properties, and the
    validator REFUSES any colour that is not in that palette. Not a
    suggestion the model may drift from: a colour outside the palette is a
    rejected drawing.

3.  **Every mark was a small pictogram on a badge.** The composition is now
    mandated and checked: a full-bleed ground rect covering the whole frame,
    a subject that fills it, few enough shapes to read at 24px, nothing
    narrower than a stroke a thumbnail can show.

**Round 25 reverses half of point 3, deliberately.** Round 20's composition
rule was "HUGE SUBJECT — it fills most of the frame and runs off at least one
edge, clipped by it. A small object centred with space around it is the
failure to avoid." That is the direction that produced the crude flat blobs
in the library today, and it produced them by working exactly as written: two
to four bold shapes, scaled until they bleed off the frame, IS a blob. Nothing
survives being cropped by the frame except the crop.

So the direction is now the opposite one, and it is measured rather than
asked for. The full-bleed GROUND stays — it is right, the OS masks the corners
and a mark on transparency reads as a sticker. What changes is the subject:
ONE glyph, centred, drawn inside a safe area (:data:`ICON_SAFE_MIN` to
:data:`ICON_SAFE_MAX` of the 96 frame, ~14.5% clear on all four sides), and
:func:`sanitize_svg` now PARSES the geometry and refuses a drawing whose
content crosses that inset, is too small to be the subject, or sits off to one
side. A prompt asking for centring on its own does nothing; the parse is the
part with teeth. Absolute path commands are required for the same reason —
a relative path cannot be bounded without interpreting the whole path, so an
un-measurable drawing is refused rather than waved through.

:data:`ICON_GENERATION` goes to 3 with that reversal: every generation-2 icon
on every volume was drawn to the bleed direction, so every one of them is a
blob and has to be redrawn.

**No two apps share a symbol.** :func:`choose_subject` is told every subject
already in use on this volume, and a returned key that collides is rejected
and asked again. The key is stored in the sidecar, so the constraint survives
restarts.

**What the model draws is refused, not stripped** — the whole file, on any
violation. A mark that had to be edited to be safe or on-palette is a mark
nobody has looked at; the refusal goes back with its reason and it is drawn
again.

**The reported source is a claim, and it is kept honest** (round 24).
``"model"`` means the mark was drawn AND persisted; ``"fallback"`` means the
three-band holding mark is on disk. A drawn mark that cannot be stored (one
retry, then :class:`~app.agent.skills.builtins.app_html.store.AppStoreError`)
RAISES out of :func:`ensure_icon` — the step must not claim a mark it does
not have, and the caller's except path reports it skipped. Every degrade
from designed mark to bands is logged at WARNING with its branch, so a fleet
where every icon degrades is legible from logs rather than silent.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

from app.agent.skills.builtins.app_html import palette as palette_mod
from app.agent.skills.builtins.app_html import store
from app.agent.skills.builtins.app_html.store import AppStoreError

logger = logging.getLogger(__name__)

ICON_DIR = ".icons"
ICON_SUFFIX = ".svg"
SIDECAR_SUFFIX = ".json"

#: Bump to redraw every icon on every volume.
#:
#: `is_stale` compares this to the sidecar, so raising it makes the whole
#: library stale at once and the list-route backfill redraws it — no
#: migration, no fleet sweep, and the same mechanism that backfilled the
#: briefs. Generation 2 was the round-20 correction: everything drawn by
#: generation 1 is a hash-coloured badge and had to go.
#:
#: Generation 3 is round 25 and costs the whole library another redraw, on
#: purpose. Generation 2 mandated a subject cropped by the frame edge (see
#: the module docstring), and a two-to-four-shape subject scaled until it
#: bleeds is the flat blob every tile currently shows. There is no icon on
#: any volume that is worth keeping under the new direction, so the staleness
#: check fails all of them at once and the list-route backfill redraws them.
ICON_GENERATION = 3

#: Names the subject; a two-line answer a cheap model gets right.
SUBJECT_MODEL = "gpt-4o-mini"
#: Draws it. Same tier deliberately — the quality came from splitting the job
#: and from the constraints, both measured, not from a bigger model, which
#: was NOT measured and so is not claimed.
DRAW_MODEL = "gpt-4o-mini"
SUBJECT_TIMEOUT_S = 20
LOGO_TIMEOUT_S = 40

#: Attempts, each one handed the previous refusal. Two was not enough: a
#: measured run fell back to a holding mark on one app in six, and the
#: refusals were single fixable faults (a stray white, a missing ground)
#: rather than the model being unable to draw the thing.
MAX_DRAW_ATTEMPTS = 3

MAX_ICON_BYTES = 24 * 1024
MIN_ICON_BYTES = 120

#: A mark is a ground plus two to four shapes. Below three there is nothing
#: on the ground; far above it, it is an illustration that will be mud at
#: 24px.
MIN_SHAPES = 3
#: Measured: a fretboard drawn with six strings, four frets and three dots
#: is thirteen shapes and reads as a lattice at any size. Twelve is the
#: point past which a mark stops being reducible.
MAX_SHAPES = 12

#: Thinner than this vanishes in a 24px tile.
MIN_STROKE_UNITS = 5.0

# ── The frame, and where in it the glyph is allowed to be (round 25) ──
#
# Round 20 mandated a subject that bleeds off the frame edge and forbade "a
# small object centred with space around it". That is reversed here (module
# docstring): the crude flat blobs in the library ARE that direction working.
# The numbers below are the reversal, and they are checked by parsing the
# drawing rather than by asking the model nicely.

#: Every mark is drawn in a 96×96 viewBox. Coordinates from any other viewBox
#: are scaled into this frame before they are measured, so a model that emits
#: ``0 0 24 24`` is judged on its composition and not on its units.
ICON_FRAME = 96.0

#: The safe area: the glyph lives inside the centred box from 14 to 82, i.e.
#: ~14.6% clear on all four sides and a glyph ~71% of the frame. The ground
#: rect is the ONLY thing that reaches the edge — an app icon is masked by the
#: OS, so a ground that stops short shows the mask, while a SUBJECT that runs
#: into the mask is the amputated blob this round removes.
ICON_SAFE_MIN = 14.0
ICON_SAFE_MAX = 82.0

#: Slack on the safe area, in frame units. Curves are measured by sampling,
#: which is a hair under the true extent, and a model asked for 14 will
#: sometimes land on 13.6. Two units still leaves a real 12-unit margin at
#: the very worst, so nothing that passes is anywhere near the edge — this
#: buys tolerance for arithmetic, not for the art direction.
ICON_SAFE_SLACK = 2.0

#: The glyph has to BE the icon. Under 40 units on its longer axis it is a
#: mark adrift in a field of ground colour — 40/96 of the frame is 10px of a
#: 24px tile, and below that there is not enough silhouette left to name.
#: Kept at 40 after looking: on the contact sheet a 40×40 glyph at 24px is
#: modest but legible, and refusing it would be tighter than the evidence.
ICON_MIN_GLYPH_LONG = 40.0
#: …and on its shorter axis. This was 18 by arithmetic (18/96 of a 24px tile
#: is 4.5px) and 18 was wrong: `scripts/icon_contact_sheet.py` renders a
#: 68×18 block at 24px and it is a dash, not an object — no silhouette, and
#: the detail inside it is gone. At 24 it becomes a shape with an inside. The
#: number moved because the sheet was looked at, which is the only way an
#: aesthetic threshold gets chosen.
ICON_MIN_GLYPH_SHORT = 24.0

#: The longer side of the BIGGEST single painted shape, at minimum.
#:
#: Round 25 follow-up. Every other size rule measures the UNION of the glyph's
#: shapes, and a union is not a mark: two 2-unit specks at opposite corners of
#: the safe box measure 68×68, dead centre, and satisfied the safe-area, size
#: and centring rules all at once while being two specks. A mark built of two
#: to four bold shapes always has one substantial piece; anything whose
#: largest element is under a quarter of the frame is confetti at 24px. Set to
#: `ICON_MIN_GLYPH_SHORT` deliberately — a shape that would be too thin to be
#: the whole glyph is also too thin to be its anchor.
ICON_MIN_LARGEST_SHAPE = ICON_MIN_GLYPH_SHORT

#: How far the glyph's bounding-box centre may sit from 48,48. Nine units is
#: 2.25px of a 24px tile — the point at which a mark stops reading as an
#: off-centre composition and starts reading as one that was placed wrong.
ICON_MAX_CENTRE_DRIFT = 9.0

#: Points sampled per curve segment when bounding a path. At icon scale the
#: worst-case shortfall against the true extent is well under a third of a
#: unit, which :data:`ICON_SAFE_SLACK` already covers many times over.
_CURVE_SAMPLES = 12

# `animate\b` did NOT match `<animateTransform`: \b between "e" and "T" is not
# a word boundary, so the whole SMIL family except bare <animate> walked
# through this gate. Same bug for `text\b` and `<textPath>`. Both are now
# prefix matches. <style> is here because a stylesheet can reach off-origin
# with @import and url(), neither of which is an href.
_FORBIDDEN_TAGS = re.compile(
    r"<\s*(script|foreignObject|iframe|embed|object|audio|video|"
    r"animate[a-z]*|set|handler|style|text[a-z]*|tspan|filter|"
    # `use` re-draws a shape defined elsewhere, at an offset of its own. The
    # geometry scan measures the definition where it is WRITTEN, so a mark
    # assembled from `<use href="#id" x=… y=…>` renders somewhere the safe-area
    # rule never looked — the one way left to place a glyph the validator
    # cannot see. A mark of two to four shapes has no need to instance
    # anything, so it is refused rather than taught about (round 25).
    r"use|"
    r"feGaussianBlur|feDropShadow)\b", re.IGNORECASE)
_EVENT_ATTR_RE = re.compile(r"""\son[a-z]+\s*=\s*["']""", re.IGNORECASE)
#: Every href/src in the document, whatever the scheme. The old pair of rules
#: asked for `//` or `data:` specifically, so `javascript:` matched neither
#: and `<a xlink:href="javascript:alert(1)">` passed. An icon has nothing to
#: link to, so the rule is now the general one: nothing but a same-document
#: `#fragment`.
_LINK_ATTR_RE = re.compile(
    r"""(?:^|[\s;])(?:xlink:href|href|src)\s*=\s*["']([^"']*)""", re.IGNORECASE)
#: CSS `url(...)`, which is a reference the href rule cannot see. `url(#g)`
#: is how a gradient fill is named and is the one allowed form.
_CSS_URL_RE = re.compile(r"""url\(\s*["']?\s*(?!#)([^)"']*)""", re.IGNORECASE)
_SVG_OPEN_RE = re.compile(r"<svg\b[^>]*>", re.IGNORECASE)
_VIEWBOX_RE = re.compile(r"\bviewBox\s*=\s*[\"']([^\"']+)[\"']", re.IGNORECASE)
_FENCE_RE = re.compile(r"^```(?:svg|xml|html)?\s*|\s*```$", re.MULTILINE)
_SHAPE_RE = re.compile(
    r"<(path|rect|circle|ellipse|polygon|polyline|line)\b", re.IGNORECASE)
_COLOUR_RE = re.compile(
    r"""(?:fill|stroke|stop-color)\s*[:=]\s*["']?\s*(#[0-9a-fA-F]{3,6}|rgb\([^)]*\)|[a-z]+)""",
    re.IGNORECASE)
_STROKE_W_RE = re.compile(r"""stroke-width\s*[:=]\s*["']?\s*([0-9.]+)""", re.IGNORECASE)

#: A drawing element and everything inside its open tag.
_ELEMENT_RE = re.compile(
    r"<(path|rect|circle|ellipse|polygon|polyline|line)\b([^>]*)>", re.IGNORECASE)
#: Containers whose contents are DEFINITIONS, not drawing. Stripped before the
#: geometry scan — see the note in `_scan`. Non-greedy and case-insensitive;
#: an unclosed container simply does not match and is measured as before,
#: which is the fail-open direction.
_DEFS_RE = re.compile(
    r"<(defs|mask|clipPath|pattern|symbol)\b[^>]*>.*?</\1\s*>",
    re.IGNORECASE | re.DOTALL)
_ATTR_RE = re.compile(
    r"""([a-zA-Z][a-zA-Z0-9:_-]*)\s*=\s*(?:"([^"]*)"|'([^']*)')""")
_DECL_RE = re.compile(r"""\s*([a-zA-Z-]+)\s*:\s*([^;]+)""")
_TRANSFORM_RE = re.compile(r"""\btransform\s*=\s*["']""", re.IGNORECASE)
_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")
#: Path command letters. Lowercase means relative, and a relative path cannot
#: be bounded without interpreting every preceding segment — so it is refused
#: and the prompt says so. `e` is absent on purpose: it is an exponent, not a
#: command.
_REL_CMD_RE = re.compile(r"[mzlhvcsqta]")
_ABS_CMD_SPLIT_RE = re.compile(r"([MZLHVCSQTA])")
#: Coordinate pairs each absolute command consumes per repetition.
_PATH_ARITY = {"M": 2, "L": 2, "T": 2, "H": 1, "V": 1,
               "C": 6, "S": 4, "Q": 4, "A": 7, "Z": 0}

#: Colour keywords a drawing may use that are not "a colour": they either
#: paint nothing or defer to something already checked.
_COLOUR_KEYWORDS = {"none", "transparent", "currentcolor", "inherit", "url"}


class IconError(Exception):
    """A drawing this module will not store."""


# ── Paths ─────────────────────────────────────────────────────────────

def icon_dir() -> str:
    return os.path.join(store.apps_root(), ICON_DIR)


def _jailed(slug: str, suffix: str) -> str:
    slug = store.normalise_slug(slug)
    root = os.path.realpath(store.apps_root())
    expected = os.path.join(root, ICON_DIR)
    full = os.path.realpath(os.path.join(expected, slug + suffix))
    if os.path.dirname(full) != expected:
        raise AppStoreError(f"refusing icon path outside the app root: {slug!r}")
    return full


def icon_path(slug: str) -> str:
    return _jailed(slug, ICON_SUFFIX)


def sidecar_path(slug: str) -> str:
    return _jailed(slug, SIDECAR_SUFFIX)


def read_icon(slug: str) -> Optional[str]:
    try:
        with open(icon_path(slug), "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except (OSError, AppStoreError):
        return None


def has_icon(slug: str) -> bool:
    """Is there a mark for this app? A stat, not a read.

    The list route asks this once per app to set a boolean, and it used to do
    it by reading the whole SVG — up to 24 KB off disk per app, decoded to
    text, to produce `True`. On a library of twenty-two apps that is half a
    megabyte of file I/O for twenty-two bits.
    """
    try:
        return os.path.getsize(icon_path(slug)) > 0
    except (OSError, AppStoreError):
        return False


def read_sidecar(slug: str) -> Dict[str, str]:
    try:
        with open(sidecar_path(slug), "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}
    except (OSError, ValueError, AppStoreError):
        return {}


def delete_icon(slug: str) -> bool:
    removed = False
    for path_of in (icon_path, sidecar_path):
        try:
            os.unlink(path_of(slug))
            removed = True
        except (OSError, AppStoreError):
            pass
    return removed


def _store_icon(slug: str, svg: str, *, source: str, title: str,
                purpose: str = "", subject: str = "",
                palette: Sequence[str] = ()) -> None:
    path = icon_path(slug)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    store.repair_permissions(os.path.dirname(path))
    store._atomic_write(path, svg.encode("utf-8"), prefix=f".icon-{slug}-")
    store._atomic_write(
        sidecar_path(slug),
        json.dumps({
            "slug": slug,
            "source": source,
            "gen": str(ICON_GENERATION),
            "title": title or slug,
            "subject": subject,
            "identity": identity_hash(title, purpose, palette),
        }, sort_keys=True).encode("utf-8"),
        prefix=f".iconmeta-{slug}-",
    )


def subjects_in_use(exclude: str = "") -> List[str]:
    """Every subject already drawn on this volume, so the next one differs."""
    out: List[str] = []
    try:
        for slug in store.read_manifest():
            if slug == exclude:
                continue
            subject = read_sidecar(slug).get("subject", "").strip()
            if subject and subject not in out:
                out.append(subject)
    except Exception:  # noqa: BLE001 - a hint that cannot be gathered is not fatal
        logger.debug("[app_html] could not list icon subjects", exc_info=True)
    return out


# ── Validation ────────────────────────────────────────────────────────

def _colours_used(svg: str) -> List[str]:
    out: List[str] = []
    for raw in _COLOUR_RE.findall(svg):
        value = raw.strip().lower()
        if value in _COLOUR_KEYWORDS or value.startswith("url"):
            continue
        if value.startswith("#"):
            value = palette_mod.normalise(value)
        if value not in out:
            out.append(value)
    return out


# ── Measuring the drawing (round 25) ──────────────────────────────────
#
# All of this exists so that "centred, inside a safe area" is a MEASUREMENT
# rather than a sentence in a prompt. The repo's own lesson from round 20 is
# that a validator with teeth beats a longer prompt: banning the stock glyph
# in words changed nothing, rejecting it changed everything. Asking for
# padding in words would go the same way.


def _bezier(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Sample a quadratic or cubic Bézier. Sampling rather than the control
    hull because the hull over-estimates a curve's extent, and an
    over-estimate here is a refusal of a drawing that was fine."""
    n = len(points) - 1
    out: List[Tuple[float, float]] = []
    for k in range(_CURVE_SAMPLES + 1):
        t = k / _CURVE_SAMPLES
        x = y = 0.0
        for i, (px, py) in enumerate(points):
            w = (math.comb(n, i) * (t ** i) * ((1 - t) ** (n - i)))
            x += w * px
            y += w * py
        out.append((x, y))
    return out


def _arc(p0: Tuple[float, float], rx: float, ry: float, rot: float,
         large: float, sweep: float,
         p1: Tuple[float, float]) -> List[Tuple[float, float]]:
    """Sample an elliptical arc. Endpoint → centre parameterisation, straight
    out of the SVG spec: the endpoints alone do NOT bound an arc — a
    semicircle between two points on the same line bulges a full radius away
    from both — and an unbounded bulge is exactly a bleed off the frame.
    """
    rx, ry = abs(rx), abs(ry)
    if rx == 0 or ry == 0 or p0 == p1:
        return [p0, p1]
    phi = math.radians(rot)
    cosp, sinp = math.cos(phi), math.sin(phi)
    dx2, dy2 = (p0[0] - p1[0]) / 2.0, (p0[1] - p1[1]) / 2.0
    x1p = cosp * dx2 + sinp * dy2
    y1p = -sinp * dx2 + cosp * dy2
    lam = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry)
    if lam > 1:
        scale = math.sqrt(lam)
        rx, ry = rx * scale, ry * scale
    den = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    if den <= 0:
        return [p0, p1]
    num = max(rx * rx * ry * ry - den, 0.0)
    coef = math.sqrt(num / den)
    if bool(large) == bool(sweep):
        coef = -coef
    cxp, cyp = coef * rx * y1p / ry, -coef * ry * x1p / rx
    cx = cosp * cxp - sinp * cyp + (p0[0] + p1[0]) / 2.0
    cy = sinp * cxp + cosp * cyp + (p0[1] + p1[1]) / 2.0

    def _angle(ux: float, uy: float, vx: float, vy: float) -> float:
        norm = math.hypot(ux, uy) * math.hypot(vx, vy)
        if norm == 0:
            return 0.0
        cosine = max(-1.0, min(1.0, (ux * vx + uy * vy) / norm))
        angle = math.acos(cosine)
        return -angle if (ux * vy - uy * vx) < 0 else angle

    ux, uy = (x1p - cxp) / rx, (y1p - cyp) / ry
    vx, vy = (-x1p - cxp) / rx, (-y1p - cyp) / ry
    theta = _angle(1.0, 0.0, ux, uy)
    delta = _angle(ux, uy, vx, vy)
    if not sweep and delta > 0:
        delta -= 2 * math.pi
    elif sweep and delta < 0:
        delta += 2 * math.pi
    out = []
    for k in range(_CURVE_SAMPLES + 1):
        t = theta + delta * k / _CURVE_SAMPLES
        out.append((cx + rx * cosp * math.cos(t) - ry * sinp * math.sin(t),
                    cy + rx * sinp * math.cos(t) + ry * cosp * math.sin(t)))
    return out


def _path_points(d: str) -> List[Tuple[float, float]]:
    """Every point an ABSOLUTE path touches. Relative commands are refused
    upstream, which is what makes this tractable without a full renderer."""
    pieces = _ABS_CMD_SPLIT_RE.split(d)
    pts: List[Tuple[float, float]] = []
    cur = start = (0.0, 0.0)
    ctrl: Optional[Tuple[float, float]] = None
    prev = ""
    open_subpath = False
    for i in range(1, len(pieces), 2):
        cmd = pieces[i]
        nums = [float(n) for n in _NUM_RE.findall(pieces[i + 1])]
        arity = _PATH_ARITY[cmd]
        if cmd == "Z":
            # Only after a moveto. A Z with nothing before it would otherwise
            # put a phantom point on the origin and refuse the whole drawing
            # for reaching 0,0 — a malformed path must not be read as a mark
            # that bleeds off the corner.
            if open_subpath:
                cur = start
                pts.append(cur)
            prev = cmd
            continue
        if arity == 0 or len(nums) < arity:
            continue
        if cmd == "M":
            open_subpath = True
        first = True
        for j in range(0, len(nums) - arity + 1, arity):
            a = nums[j:j + arity]
            # Repeated arguments after an M are an implicit L (SVG 1.1 8.3.2),
            # and a reflected control point only reflects after a matching
            # curve — both of which decide where the next segment starts, so
            # both have to be right or every later coordinate is fiction.
            step = "L" if (cmd == "M" and not first) else cmd
            if step == "M":
                cur = start = (a[0], a[1])
                pts.append(cur)
            elif step == "L":
                cur = (a[0], a[1])
                pts.append(cur)
            elif step == "H":
                cur = (a[0], cur[1])
                pts.append(cur)
            elif step == "V":
                cur = (cur[0], a[0])
                pts.append(cur)
            elif step == "C":
                c1, c2, end = (a[0], a[1]), (a[2], a[3]), (a[4], a[5])
                pts.extend(_bezier([cur, c1, c2, end]))
                ctrl, cur = c2, end
            elif step == "S":
                mirror = ((2 * cur[0] - ctrl[0], 2 * cur[1] - ctrl[1])
                          if ctrl and prev in ("C", "S") else cur)
                c2, end = (a[0], a[1]), (a[2], a[3])
                pts.extend(_bezier([cur, mirror, c2, end]))
                ctrl, cur = c2, end
            elif step == "Q":
                c1, end = (a[0], a[1]), (a[2], a[3])
                pts.extend(_bezier([cur, c1, end]))
                ctrl, cur = c1, end
            elif step == "T":
                mirror = ((2 * cur[0] - ctrl[0], 2 * cur[1] - ctrl[1])
                          if ctrl and prev in ("Q", "T") else cur)
                end = (a[0], a[1])
                pts.extend(_bezier([cur, mirror, end]))
                ctrl, cur = mirror, end
            elif step == "A":
                end = (a[5], a[6])
                pts.extend(_arc(cur, a[0], a[1], a[2], a[3], a[4], end))
                ctrl, cur = None, end
            first = False
            prev = step
    return pts


def _attributes(chunk: str) -> Dict[str, str]:
    """An element's attributes, with any ``style="a:b;c:d"`` folded in so a
    declaration and an attribute are the same thing to every rule here."""
    out: Dict[str, str] = {}
    for name, dq, sq in _ATTR_RE.findall(chunk):
        out[name.lower()] = dq if dq else sq
    for prop, value in _DECL_RE.findall(out.get("style", "")):
        out[prop.lower()] = value.strip()
    return out


def _length(attrs: Dict[str, str], name: str, span: float,
            default: float = 0.0) -> float:
    raw = (attrs.get(name) or "").strip()
    if not raw:
        return default
    match = _NUM_RE.match(raw)
    if not match:
        return default
    value = float(match.group(0))
    return value * span / 100.0 if raw.rstrip().endswith("%") else value


def _element_box(tag: str, attrs: Dict[str, str], vb_w: float,
                 vb_h: float) -> Optional[Tuple[float, float, float, float]]:
    """One shape's bounding box in the drawing's own user units."""
    tag = tag.lower()
    if tag == "rect":
        x = _length(attrs, "x", vb_w)
        y = _length(attrs, "y", vb_h)
        w = _length(attrs, "width", vb_w)
        h = _length(attrs, "height", vb_h)
        return (x, y, x + w, y + h) if w > 0 and h > 0 else None
    if tag == "circle":
        cx, cy = _length(attrs, "cx", vb_w), _length(attrs, "cy", vb_h)
        r = _length(attrs, "r", vb_w)
        return (cx - r, cy - r, cx + r, cy + r) if r > 0 else None
    if tag == "ellipse":
        cx, cy = _length(attrs, "cx", vb_w), _length(attrs, "cy", vb_h)
        rx, ry = _length(attrs, "rx", vb_w), _length(attrs, "ry", vb_h)
        return (cx - rx, cy - ry, cx + rx, cy + ry) if rx > 0 and ry > 0 else None
    if tag == "line":
        xs = [_length(attrs, "x1", vb_w), _length(attrs, "x2", vb_w)]
        ys = [_length(attrs, "y1", vb_h), _length(attrs, "y2", vb_h)]
        return (min(xs), min(ys), max(xs), max(ys))
    if tag in ("polygon", "polyline"):
        nums = [float(n) for n in _NUM_RE.findall(attrs.get("points", ""))]
        pts = list(zip(nums[0::2], nums[1::2]))
    elif tag == "path":
        pts = _path_points(attrs.get("d", ""))
    else:  # pragma: no cover - _ELEMENT_RE cannot produce anything else
        return None
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def _fill(attrs: Dict[str, str]) -> str:
    """An element's fill, normalised, or ``""`` when it has none this module
    can compare (a gradient, a keyword, an omitted attribute)."""
    value = (attrs.get("fill") or "").strip().lower()
    return palette_mod.normalise(value) if value.startswith("#") else ""


def _paints_nothing(attrs: Dict[str, str]) -> bool:
    """True when this element puts no ink on the frame at all.

    ``fill="none"`` with no stroke is a spacer or a construction line. It has
    geometry, so it measures like a shape, and before round 25's follow-up it
    counted toward the glyph's extent — which is how an invisible 68×68 rect
    could carry a 3-unit dot past the safe-area, size AND centring rules in
    one go.
    """
    fill = (attrs.get("fill") or "").strip().lower()
    if fill not in ("none", "transparent"):
        return False
    return _painted_extent(attrs) <= 0.0


def _painted_extent(attrs: Dict[str, str]) -> float:
    """Half the stroke this element paints with, which its geometry does not
    include. A centreline at 14 with a 10-unit stroke actually reaches 9."""
    stroke = (attrs.get("stroke") or "").strip().lower()
    if not stroke or stroke in ("none", "transparent"):
        return 0.0
    try:
        width = float((attrs.get("stroke-width") or "1").strip().rstrip("%"))
    except ValueError:
        width = 1.0
    return max(width, 0.0) / 2.0


Box = Tuple[float, float, float, float]


def _scan(text: str) -> Optional[Tuple[bool, List[Box]]]:
    """``(a full-bleed ground is present, boxes of everything else)``.

    Every box is in FRAME units — a drawing that declares another viewBox is
    scaled into the 96 frame first, so it is judged on its composition rather
    than on its units. ``None`` when the drawing cannot be measured at all,
    which callers must treat as "no opinion": every rule built on this is
    about how the mark LOOKS, and the module docstring's contract is that a
    look rule never turns into a build failure on a parse it did not manage.
    """
    # Round 25 follow-up. Shapes inside `<defs>`, `<mask>`, `<clipPath>`,
    # `<pattern>` or `<symbol>` are DEFINITIONS: they are not painted where
    # they are written, and some are never painted at all. Measuring them was
    # wrong in both directions — a 4-unit rect at 0,0 inside a `<defs>`
    # refused an otherwise perfectly centred mark, and a full-frame rect
    # inside a `<mask>` satisfied the full-bleed-ground check for a drawing
    # that had no ground at all. Neither is reachable by an attacker; both are
    # ordinary things a model writes, and the false refusal is the expensive
    # one — three of those and the app falls back to the plain holding bands.
    text = _DEFS_RE.sub(" ", text)
    open_tag = _SVG_OPEN_RE.search(text)
    vb = _VIEWBOX_RE.search(open_tag.group(0)) if open_tag else None
    if not vb:
        return None
    parts = [float(n) for n in _NUM_RE.findall(vb.group(1))]
    if len(parts) != 4 or parts[2] <= 0 or parts[3] <= 0:
        return None
    min_x, min_y, vb_w, vb_h = parts
    sx, sy = ICON_FRAME / vb_w, ICON_FRAME / vb_h

    ground = False
    ground_fill = ""
    boxes: List[Box] = []
    for tag, chunk in _ELEMENT_RE.findall(text):
        attrs = _attributes(chunk)
        box = _element_box(tag, attrs, vb_w, vb_h)
        if box is None:
            continue
        pad = _painted_extent(attrs)
        x0 = (box[0] - pad - min_x) * sx
        y0 = (box[1] - pad - min_y) * sy
        x1 = (box[2] + pad - min_x) * sx
        y1 = (box[3] + pad - min_y) * sy
        if (tag.lower() == "rect" and x0 <= 0.5 and y0 <= 0.5
                and x1 >= ICON_FRAME - 0.5 and y1 >= ICON_FRAME - 0.5):
            # The full-bleed ground, by design — the one thing allowed to
            # reach the edge, and never part of the glyph's extent.
            ground = True
            ground_fill = ground_fill or _fill(attrs)
            continue
        if _paints_nothing(attrs):
            # Round 25 follow-up. A shape with no fill and no stroke is a
            # spacer, not a mark. Counting it let an invisible 68×68 rect
            # carry a 3-unit dot through every composition rule at once: the
            # dot alone is far too small and far off centre, and the rect
            # made the pair look like a properly sized, properly centred
            # glyph. Same family as the knockout exemption below — a shape is
            # measured only where it actually puts ink.
            continue
        if ground_fill and _fill(attrs) == ground_fill and not _painted_extent(attrs):
            # A KNOCKOUT: a shape in the ground's own colour, which is how a
            # crescent, a cut, a bite or a gap gets drawn. It paints nothing
            # visible, so where it runs is not where the mark runs — refusing
            # it for crossing the safe area would push the model away from
            # the one technique that makes a two-shape subject look drawn.
            continue
        boxes.append((x0, y0, x1, y1))
    return ground, boxes


def _safe_scan(text: str) -> Optional[Tuple[bool, List[Box]]]:
    """:func:`_scan`, but a surprise in the parser is "no opinion" rather than
    an exception.

    `draw_mark` catches :class:`IconError` and nothing else, so a TypeError
    raised in here on some shape nobody anticipated would escape the retry
    loop. `skill.py` does catch bare `Exception`, so it would not fail the
    BUILD — the cost is narrower and still not worth paying: the app loses its
    mark entirely for that run instead of degrading to the holding bands,
    because the fallback is written after the point that would have thrown."""
    try:
        return _scan(text)
    except Exception:  # noqa: BLE001 - deliberate, see above
        logger.debug("[app_html] could not measure an icon", exc_info=True)
        return None


def measure_glyph(text: str) -> Optional[Box]:
    """The bounding box of everything that is NOT the ground, in frame units,
    or ``None`` when there is nothing measurable."""
    scan = _safe_scan(text)
    if not scan or not scan[1]:
        return None
    boxes = scan[1]
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _check_composition(text: str) -> None:
    """Round 25's art direction, as three measurements. Raises
    :class:`IconError` with a reason written to be handed back for a redraw."""
    box = measure_glyph(text)
    if box is None:
        return
    x0, y0, x1, y1 = box
    low, high = ICON_SAFE_MIN - ICON_SAFE_SLACK, ICON_SAFE_MAX + ICON_SAFE_SLACK
    if x0 < low or y0 < low or x1 > high or y1 > high:
        raise IconError(
            f"the subject runs from {x0:.0f},{y0:.0f} to {x1:.0f},{y1:.0f} and "
            f"leaves the safe area. Everything except the ground rect must sit "
            f"inside the centred box from {ICON_SAFE_MIN:.0f} to "
            f"{ICON_SAFE_MAX:.0f} on both axes — clear of all four edges, "
            f"nothing cropped by the frame. Scale the subject down and move it "
            f"back to the middle; remember a stroke paints half its width "
            f"outside the line")

    width, height = x1 - x0, y1 - y0
    if max(width, height) < ICON_MIN_GLYPH_LONG or min(width, height) < ICON_MIN_GLYPH_SHORT:
        raise IconError(
            f"the subject is only {width:.0f}×{height:.0f} in a 96 frame — a "
            f"small mark adrift in the ground colour. Grow it to fill the safe "
            f"box from {ICON_SAFE_MIN:.0f} to {ICON_SAFE_MAX:.0f}: at least "
            f"{ICON_MIN_GLYPH_LONG:.0f} units across its longer side and "
            f"{ICON_MIN_GLYPH_SHORT:.0f} across its shorter one")

    # The union is not the mark. See `ICON_MIN_LARGEST_SHAPE`: every rule
    # above is satisfied by two specks in opposite corners of the safe box.
    scan = _safe_scan(text)
    boxes = scan[1] if scan else []
    if boxes:
        biggest = max(max(b[2] - b[0], b[3] - b[1]) for b in boxes)
        if biggest < ICON_MIN_LARGEST_SHAPE:
            raise IconError(
                f"the biggest shape in the mark is only {biggest:.0f} units "
                f"across — the subject is made of specks, which is a smudge at "
                f"24px however well they are spread out. Draw it as two to "
                f"four BOLD shapes, at least one of them spanning most of the "
                f"safe box from {ICON_SAFE_MIN:.0f} to {ICON_SAFE_MAX:.0f}")

    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    drift = max(abs(cx - ICON_FRAME / 2), abs(cy - ICON_FRAME / 2))
    if drift > ICON_MAX_CENTRE_DRIFT:
        raise IconError(
            f"the subject's centre is at {cx:.0f},{cy:.0f} instead of 48,48 — "
            f"it sits off to one side. Centre it: the same amount of ground "
            f"colour above and below it, and the same to left and right")


def sanitize_svg(raw: str, palette: Optional[Sequence[str]] = None) -> str:
    """Return the drawing, or raise :class:`IconError` naming what is wrong.

    Refuses rather than strips, and the message is written to be handed
    straight back to the model as the reason for a redraw. Everything checked
    here is something that made a batch of icons look generated: a floating
    badge, a colour from nowhere, detail that dies at 24px — and, since round
    25, a subject cropped by the frame, adrift in it, or shoved to one side,
    which are measured off the parsed geometry rather than asked for.

    ``palette`` is optional so the structural rules can be applied to a
    drawing whose app is unknown (a fallback, a test); when it is given, the
    colour rule is enforced and it is the strictest rule in the file.
    """
    text = _FENCE_RE.sub("", (raw or "").strip()).strip()
    if not text:
        raise IconError("the icon is empty")
    start = text.lower().find("<svg")
    if start == -1:
        raise IconError("that is not an SVG — it must start with <svg")
    end = text.lower().rfind("</svg>")
    if end == -1:
        raise IconError("the SVG is not closed — it must end with </svg>")
    text = text[start:end + len("</svg>")]

    data = text.encode("utf-8")
    if len(data) < MIN_ICON_BYTES:
        raise IconError("the icon is a stub — draw the actual mark")
    if len(data) > MAX_ICON_BYTES:
        raise IconError(
            f"the icon is {len(data)} bytes, over the {MAX_ICON_BYTES} byte "
            f"limit — a mark is a few bold shapes, not a traced photograph")

    if _FORBIDDEN_TAGS.search(text):
        raise IconError(
            "the icon contains script, text or an embedded element — paths, "
            "shapes and gradients only, and never <text>: an icon that has to "
            "be read is not an icon")
    if _EVENT_ATTR_RE.search(text):
        raise IconError("the icon has an on… event attribute — an icon does not run")
    for ref in _LINK_ATTR_RE.findall(text):
        target = ref.strip()
        if target and not target.startswith("#"):
            raise IconError(
                f"the icon points at {target[:60]!r} — an icon links to "
                f"nothing and loads nothing. Only a same-document #fragment "
                f"is allowed; it must be a self-contained vector drawing")
    for ref in _CSS_URL_RE.findall(text):
        raise IconError(
            f"the icon loads {ref.strip()[:60]!r} through a CSS url() — the "
            f"only url() an icon may use is url(#id) naming a gradient it "
            f"defines itself")
    open_tag = _SVG_OPEN_RE.search(text)
    if not open_tag or not _VIEWBOX_RE.search(open_tag.group(0)):
        raise IconError("the <svg> has no viewBox — without one it cannot scale")

    shapes = _SHAPE_RE.findall(text)
    if len(shapes) < MIN_SHAPES:
        raise IconError(
            f"only {len(shapes)} shape(s) — that is a ground with nothing on "
            f"it. Build the subject from two to four overlapping shapes on "
            f"top of the full-bleed ground")
    if len(shapes) > MAX_SHAPES:
        raise IconError(
            f"{len(shapes)} shapes is an illustration, not a mark — it will be "
            f"mud at 24px. Reduce the subject to two to four bold shapes")

    if _TRANSFORM_RE.search(text):
        # Round 25. Not a safety rule — a measurability one. A transformed
        # shape's attributes say nothing about where it lands, so the safe
        # area below would be checking coordinates the renderer never uses.
        raise IconError(
            "the icon uses a transform= — write the coordinates out instead. "
            "Where the mark sits in the frame is measured, and a transformed "
            "shape cannot be measured")
    for element, chunk in _ELEMENT_RE.findall(text):
        if element.lower() != "path":
            continue
        if _REL_CMD_RE.search(_attributes(chunk).get("d", "")):
            raise IconError(
                "the path uses relative commands (m, l, c, q, s, t, a, z) — "
                "every command must be the ABSOLUTE, capital form: M, L, C, "
                "Q, S, T, A, Z, with every coordinate written out in the 96 "
                "frame. A relative path cannot be measured against the frame, "
                "and this mark's placement in the frame is measured")

    # The ground: parsed, not pattern-matched. The old regex required width
    # BEFORE height in the source, so a perfectly good
    # `<rect height="96" width="96" .../>` was refused and burned a redraw.
    scan = _safe_scan(text)
    if scan is not None and not scan[0]:
        raise IconError(
            'there is no full-bleed ground — the FIRST element must be '
            '<rect width="96" height="96" fill="…"/> covering the whole frame. '
            'A mark floating on transparency reads as a sticker, not an icon')

    for width in _STROKE_W_RE.findall(text):
        try:
            if float(width) < MIN_STROKE_UNITS:
                raise IconError(
                    f"stroke-width {width} disappears at 24px — nothing thinner "
                    f"than {MIN_STROKE_UNITS:g} units, and prefer filled shapes "
                    f"to strokes")
        except ValueError:
            continue

    _check_composition(text)

    if palette:
        allowed = {palette_mod.normalise(c) for c in palette}
        stray = [c for c in _colours_used(text) if c not in allowed]
        if stray:
            raise IconError(
                f"uses {', '.join(stray[:4])}, which {'is' if len(stray) == 1 else 'are'} "
                f"not in this app's palette. Use ONLY these, exactly: "
                f"{', '.join(palette)}")
    return text


# ── The always-available mark ─────────────────────────────────────────

def fallback_icon(slug: str, title: str = "",
                  palette: Optional[Sequence[str]] = None) -> str:
    """A holding mark, in the APP'S colours, for a container with no model.

    Deliberately not a logo and not pretending to be one: three plain bands
    of the app's own palette. It exists so a card is never a broken image,
    it records itself as `source: "fallback"` — which :func:`is_stale` treats
    as always stale — and the first run that can reach a model replaces it.

    The colours come from the app, never from a hash of the slug. That ramp
    is the whole reason this file was rewritten: a tile whose colour has no
    relationship to the app it opens is worse than a plain one.

    Round 25 pulls the bands inside the safe area and off the edges. Two
    reasons, and the second is the one that matters: the floor has to PASS
    the validator the designed marks are held to — a spec its own fallback
    violates is a spec that has never been checked end to end — and the app's
    accent now gets a band, because :func:`palette.roles` stopped handing the
    middle colour by luminance to the subject.
    """
    colours = list(palette or [])
    if not colours:
        # No palette at all: neutral, and obviously provisional. Not a
        # generated hue — an app with no colours of its own does not acquire
        # some here.
        colours = ["#232329", "#3A3A44", "#8A8A96"]
    parts = palette_mod.roles(colours)
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96" '
        'width="96" height="96" role="img">'
        f'<rect width="96" height="96" fill="{parts.ground}"/>'
        f'<rect x="14" y="22" width="68" height="16" fill="{parts.glyph}"/>'
        f'<rect x="14" y="44" width="68" height="16" fill="{parts.detail}"/>'
        f'<rect x="14" y="66" width="42" height="16" fill="{parts.glyph}"/>'
        '</svg>'
    )


# ── The designed mark ─────────────────────────────────────────────────

_SUBJECT_SYSTEM = (
    "You choose what an app's icon should DEPICT. You do not draw it.\n"
    "\n"
    "Answer with exactly two lines and nothing else:\n"
    "KEY: <two or three words naming the object, lowercase>\n"
    "SCENE: <one sentence describing the picture, concrete enough to draw>\n"
    "\n"
    "Rules for choosing:\n"
    " - Name an object from the app's SUBJECT MATTER — the thing the app is "
    "about out in the world. A gym tracker is a dumbbell, not a tick; a "
    "recipe book is a pot, not a list; a sleep aid is a moon, not a toggle. "
    "The app's own UI is never the subject.\n"
    " - Pick the most SPECIFIC object in the app, not the category it belongs "
    "to. The test: if the object would suit ten other apps, it is wrong.\n"
    " - A habit or streak app is the hardest case and the one that goes wrong "
    "most: the completion mark is not the subject, the HABIT is. Draw the "
    "thing being tracked — the dumbbell, the glass of water, the running "
    "shoe — never the tick beside it.\n"
    " - Prefer the thing the person physically does or sees IN the app over an "
    "emblem of its topic. A pomodoro timer is a tomato cut by a time wedge, "
    "not a clock. A budget app is a stack of coins with one lifted out, not a "
    "document. A snake game is the snake's own body turning a corner toward "
    "the pellet, not a screen.\n"
    " - BANNED, however apt they feel: a clock or watch face, a sheet of paper "
    "or document, a spreadsheet or grid of cells, a map pin, a gear, a "
    "lightbulb, a magnifying glass, a checkmark, a generic app window, a bar "
    "chart, a letter or monogram.\n"
    " - It must not repeat anything in ALREADY USED. If your idea is close to "
    "one of those, choose a different aspect of this app.\n"
    " - It must be drawable as two to four bold overlapping shapes."
)

_DRAW_SYSTEM = (
    "You draw one app icon as SVG, from a scene you are given. You do not "
    "choose the subject — it has been chosen. Draw THAT.\n"
    "\n"
    "OUTPUT\n"
    " - The SVG and nothing else. No prose, no markdown fences.\n"
    ' - <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96" '
    'width="96" height="96"> … </svg>\n'
    " - Paths, shapes and gradients only. No <text>, no <image>, no <style>, "
    "no <foreignObject>, no script, no animation, no external, data: or "
    "javascript: references, no filters, no blurs, no drop shadows.\n"
    " - Under 6 KB.\n"
    " - ABSOLUTE path commands only — M, L, H, V, C, S, Q, T, A, Z, all "
    "capitals. A lowercase (relative) command is rejected, and so is any "
    "transform=. Where the mark sits in the frame is MEASURED, and neither "
    "of those can be measured.\n"
    "\n"
    "WHAT YOU ARE MAKING\n"
    "A MARK — not an illustration and not a UI pictogram. The kind of thing "
    "that works as an enamel pin or a three-colour screen print: bold, "
    "reduced, confident, recognisable across a room.\n"
    "\n"
    "COMPOSITION — all three, every time, and the first two are MEASURED\n"
    ' 1. FULL-BLEED GROUND. The FIRST element is <rect width="96" height="96" '
    'fill="…"/> and it covers the whole frame. No margin, no border, no '
    "rounded corner, no transparency. The client rounds the corners; you must "
    "not. It is the ONLY thing allowed to touch an edge.\n"
    " 2. ONE GLYPH, CENTRED, INSIDE THE SAFE AREA. Every other shape you draw "
    "must sit inside the box from 14,14 to 82,82 — a clear margin of 14 units "
    "on all four sides — and the subject must FILL that box, not float in it: "
    "at least 40 units across, and its centre on 48,48. Nothing is cropped by "
    "the frame; nothing runs off an edge. Remember a stroke paints half its "
    "width outside the line it follows, so keep stroked centrelines well "
    "inside 14–82. Think of a coin or a badge stamped in the middle of a "
    "card, not a photograph cropped to a square.\n"
    " 3. TWO TO FOUR SHAPES making ONE object, each overlapping another, so "
    "the mark reads as layered rather than flat. Reduce the scene until it is "
    "that few. Two to four shapes STRETCHED to the frame edge is a blob — "
    "reduce the object, then place it whole.\n"
    "\n"
    "IT HAS TO SURVIVE 24 PIXELS\n"
    " - Nothing narrower than 8 units; no stroke-width under 5. Prefer filled "
    "shapes to strokes. No hairlines, no outline round everything, no repeated "
    "fine detail.\n"
    " - If a detail would vanish at 24px, DELETE it rather than shrink it.\n"
    " - The silhouette alone must be recognisable with all colour removed.\n"
    " - NEVER render the subject as a regular grid, a lattice or a set of "
    "repeating parallel lines. If the object really is one — a fretboard, a "
    "net, a keyboard — draw the PART of it that has a silhouette, cropped "
    "huge, not the repeating structure.\n"
    "\n"
    "COLOUR — the strictest rule here\n"
    " - Use ONLY the palette given, exactly those hex values. Any other colour "
    "is rejected outright. It is the app's own palette and the mark has to "
    "belong to the app.\n"
    " - You are told which value is the GROUND, which is the SUBJECT and "
    "which is the DETAIL. Use them in those parts. A linear gradient between "
    "two of the given values is allowed; inventing a colour is not."
)


def identity_hash(title: str, purpose: str, palette: Sequence[str] = ()) -> str:
    """What the icon DEPICTS and what it is PAINTED IN, as a stable key.

    Round 21, item 1. Until now this was title + purpose only, which made an
    icon a fact about the app's *description*: an edit that repainted the
    whole app — new ground, new accent, a different type of thing on screen —
    left the tile in last week's colours, and the mark stopped belonging to
    the app it stands for. The palette is read from the app's own CSS
    (:mod:`palette`), so it moves exactly when the app's look moves.

    It is still not "redraw on every edit" in the wasteful sense: a padding
    change, a copy fix or a bug fix leaves title, purpose and palette alone,
    so the identity is unchanged, the existing mark is kept, and the tile does
    not flicker between revisions. What changed is that "the app looks
    different now" is finally one of the things that counts as different.
    """
    colours = ",".join(palette_mod.normalise(c) for c in (palette or ()))
    basis = (
        f"{' '.join((title or '').split())}"
        f"|{' '.join((purpose or '').split())[:200]}"
        f"|{colours}"
    )
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


#: Back-compat for callers/tests written against the first version.
subject_hash = identity_hash


def icon_etag(slug: str) -> str:
    """A validator for this app's mark, or ``""`` when it has none.

    The same value the ``/artifacts/{slug}/icon`` route sends as its ETag —
    computed from the bytes, so a redraw changes it and nothing else does. It
    rides :func:`steps.artifact_payload` so a client can cache the SVG under
    (slug, etag) indefinitely and re-fetch on the one event that matters.
    """
    svg = read_icon(slug)
    if not svg:
        return ""
    return hashlib.sha256(svg.encode("utf-8")).hexdigest()[:32]


def is_stale(slug: str, *, title: str, purpose: str,
             palette: Sequence[str] = ()) -> bool:
    """Does this app need a (re)drawn icon?"""
    if read_icon(slug) is None:
        return True
    meta = read_sidecar(slug)
    if meta.get("source") != "model":
        # A holding mark is provisional by definition; upgrade it the first
        # time a model is reachable.
        return True
    if meta.get("gen") != str(ICON_GENERATION):
        # Drawn by an older art direction. Generation 1 is the hash-coloured
        # badge this rewrite exists to remove, so every one of them is stale.
        return True
    return meta.get("identity") != identity_hash(title, purpose, palette)


async def _ask(system: str, user: str, *, model: str, max_tokens: int,
               timeout: int, user_id: str, operation: str) -> Optional[str]:
    try:
        from app.services.internal_llm import call_system_llm
        return await asyncio.wait_for(
            call_system_llm(
                user_id=user_id or "", operation_type=operation, model=model,
                max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": user}], timeout=timeout,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning("[app_html] %s timed out", operation)
        return None
    except Exception:  # noqa: BLE001 - a missing model must not fail a build
        logger.warning("[app_html] %s could not run", operation, exc_info=True)
        return None


_KEY_RE = re.compile(r"KEY:\s*(.+)")
_SCENE_RE = re.compile(r"SCENE:\s*(.+)")

#: Stock subjects, checked after the model answers. The ban is in the prompt
#: too; this is the half that does not depend on the prompt being obeyed.
_BANNED_SUBJECTS = re.compile(
    r"\b(clock|watch face|document|sheet of paper|spreadsheet|grid|map pin|"
    r"gear|cog|lightbulb|light bulb|magnifying glass|checkmark|check mark|"
    r"bar chart|monogram|letter|app window)\b", re.IGNORECASE)


async def choose_subject(*, user_id: str, title: str, purpose: str,
                         used: Sequence[str]) -> Tuple[str, str]:
    """``(key, scene)`` — what this app's icon should depict. ``("", "")`` on
    failure, which the caller must treat as "do not draw", never as a default.
    """
    raw = await _ask(
        _SUBJECT_SYSTEM,
        f'App name: "{title}".\n'
        f"What it is: {' '.join((purpose or '').split())[:400] or title}\n"
        f"ALREADY USED: {', '.join(used) or '(none yet)'}\n"
        "Choose its icon's subject.",
        model=SUBJECT_MODEL, max_tokens=200, timeout=SUBJECT_TIMEOUT_S,
        user_id=user_id, operation="system.app_html.icon_subject",
    )
    if not raw:
        return "", ""
    key_m, scene_m = _KEY_RE.search(raw), _SCENE_RE.search(raw)
    key = (key_m.group(1).strip().lower()[:60] if key_m else "")
    scene = (scene_m.group(1).strip()[:400] if scene_m else "")
    if not key or not scene:
        logger.warning("[app_html] icon subject was not two lines: %r", raw[:160])
        return "", ""
    if _BANNED_SUBJECTS.search(key):
        logger.warning("[app_html] icon subject %r is a stock glyph — refused", key)
        return "", ""
    if key in {u.lower() for u in used}:
        logger.warning("[app_html] icon subject %r is already in use — refused", key)
        return "", ""
    return key, scene


async def draw_mark(*, user_id: str, title: str, scene: str,
                    palette: Sequence[str]) -> Optional[str]:
    """The SVG for a scene, validated. None if it could not be drawn."""
    # Round 25: the roles come from `palette.roles`, not from luminance
    # order. The subject is painted in the app's most CHROMATIC colour that
    # is legible on the ground — before this, "a mid colour" meant whichever
    # colour happened to sort in the middle, which on a dark app was `--muted`
    # and left the accent out of the app's own icon entirely.
    parts = palette_mod.roles(list(palette))
    brief = (
        f'App name: "{title}".\n'
        f"SCENE TO DRAW: {scene}\n"
        f"The app's palette, darkest first: {', '.join(parts.ranked)}.\n"
        f"GROUND (the full-bleed rect): {parts.ground}. "
        f"SUBJECT (the glyph itself): {parts.glyph}. "
        f"DETAIL (the one thing that must pop, drawn on the subject): "
        f"{parts.detail}. Use no other colours.\n"
        "Draw it, centred in the safe area. SVG only."
    )
    complaint = ""
    for attempt in range(MAX_DRAW_ATTEMPTS):
        raw = await _ask(
            _DRAW_SYSTEM, brief + complaint, model=DRAW_MODEL, max_tokens=1800,
            timeout=LOGO_TIMEOUT_S, user_id=user_id,
            operation="system.app_html.icon",
        )
        if not raw:
            return None
        try:
            return sanitize_svg(raw, parts.ranked)
        except IconError as exc:
            logger.info("[app_html] icon attempt %d refused: %s", attempt + 1, exc)
            # Hand the refusal back verbatim. It is written for this reader.
            complaint = (f"\n\nYour previous attempt was REJECTED: {exc}\n"
                         f"Draw it again, fixing exactly that.")
    return None


async def ensure_icon(
    slug: str, *, title: str, purpose: str = "", user_id: str = "",
    html: str = "", allow_model: bool = True,
) -> Tuple[str, str]:
    """The app's mark, drawing it if it is missing or stale.

    Returns ``(svg, source)`` where source is ``kept`` | ``model`` |
    ``fallback``. Raises :class:`AppStoreError` in exactly one case: a mark
    was drawn but could not be stored, even on a retry — returning ``model``
    there would claim a mark that is not on disk (module docstring, the
    honesty contract).

    ``html`` is the app's own source, and it is what the palette is read
    from. Without it the mark can still be drawn, but on the neutral holding
    palette — so callers that have the file should always pass it.
    """
    slug = store.normalise_slug(slug)
    # The palette is read BEFORE the staleness check, not after it: since
    # round 21 the app's own colours are part of what the mark is FOR, so
    # "is this mark still right" cannot be answered without them. (It is a
    # regex over the file the caller usually already holds — cheaper than the
    # `os.stat` the check does anyway.)
    if not html:
        try:
            html = store.read_app(slug)
        except (OSError, AppStoreError):
            html = ""
    colours = palette_mod.extract(html) if html else []

    if not is_stale(slug, title=title, purpose=purpose, palette=colours):
        existing = read_icon(slug)
        if existing:
            return existing, "kept"

    degrade = ""
    if allow_model:
        from app.agent.skills.builtins.app_html import vision
        if vision.can_call_model():
            used = subjects_in_use(exclude=slug)
            key, scene = await choose_subject(
                user_id=user_id, title=title, purpose=purpose, used=used,
            )
            if not scene:
                # One more sample before giving up the designed mark: every
                # subject failure (no answer, wrong shape, banned, duplicate)
                # is a property of one draw from the model, and choose_subject
                # has no retry of its own the way draw_mark does.
                key, scene = await choose_subject(
                    user_id=user_id, title=title, purpose=purpose, used=used,
                )
            if scene and colours:
                svg = await draw_mark(user_id=user_id, title=title, scene=scene,
                                      palette=colours)
                if svg:
                    try:
                        _store_icon(slug, svg, source="model", title=title,
                                    purpose=purpose, subject=key,
                                    palette=colours)
                    except (OSError, AppStoreError):
                        logger.warning("[app_html] could not store the icon for "
                                       "%s — retrying once", slug, exc_info=True)
                        try:
                            _store_icon(slug, svg, source="model", title=title,
                                        purpose=purpose, subject=key,
                                        palette=colours)
                        except (OSError, AppStoreError) as exc:
                            # "model" is a claim the mark is ON DISK. The
                            # caller's except path (skill.py) turns this into
                            # the honest skipped step instead of "drew a
                            # fresh mark" over an empty directory.
                            raise AppStoreError(
                                f"drew a mark for {slug} but could not store "
                                f"it") from exc
                    return svg, "model"
                degrade = f"draw refused all {MAX_DRAW_ATTEMPTS} attempts"
            elif scene and not colours:
                # No palette means no way to make the mark belong to the app,
                # and a mark in invented colours is the defect this round
                # removed. Better a holding mark that says so.
                degrade = "app has no palette of its own"
            else:
                degrade = "no usable subject, twice"

    existing = read_icon(slug)
    meta = read_sidecar(slug)
    if existing and meta.get("source") == "model" and meta.get("gen") == str(ICON_GENERATION):
        # Keep a current designed mark rather than downgrading it because one
        # generation failed.
        if degrade:
            logger.warning("[app_html] icon redraw for %s failed — %s; "
                           "keeping the current mark", slug, degrade)
        return existing, "kept"

    svg = fallback_icon(slug, title, colours)
    if degrade:
        logger.warning("[app_html] icon for %s degraded to the holding bands "
                       "— %s", slug, degrade)
    try:
        _store_icon(slug, svg, source="fallback", title=title, purpose=purpose,
                    palette=colours)
    except (OSError, AppStoreError):
        logger.warning("[app_html] could not store the holding mark for %s",
                       slug, exc_info=True)
    return svg, "fallback"
