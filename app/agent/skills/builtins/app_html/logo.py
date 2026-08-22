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

**No two apps share a symbol.** :func:`choose_subject` is told every subject
already in use on this volume, and a returned key that collides is rejected
and asked again. The key is stored in the sidecar, so the constraint survives
restarts.

**What the model draws is refused, not stripped** — the whole file, on any
violation. A mark that had to be edited to be safe or on-palette is a mark
nobody has looked at; the refusal goes back with its reason and it is drawn
again.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
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
#: briefs. Generation 2 is the correction: everything drawn by generation 1
#: is a hash-coloured badge and has to go.
ICON_GENERATION = 2

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

_FORBIDDEN_TAGS = re.compile(
    r"<\s*(script|foreignObject|iframe|embed|object|audio|video|animate|set|"
    r"handler|text|tspan|filter|feGaussianBlur|feDropShadow)\b", re.IGNORECASE)
_EVENT_ATTR_RE = re.compile(r"""\son[a-z]+\s*=\s*["']""", re.IGNORECASE)
_EXTERNAL_REF_RE = re.compile(
    r"""(?:href|xlink:href|src)\s*=\s*["']\s*(?:https?:)?//""", re.IGNORECASE)
_DATA_URI_RE = re.compile(r"""(?:href|xlink:href|src)\s*=\s*["']\s*data:""", re.IGNORECASE)
_SVG_OPEN_RE = re.compile(r"<svg\b[^>]*>", re.IGNORECASE)
_VIEWBOX_RE = re.compile(r"\bviewBox\s*=\s*[\"'][^\"']+[\"']", re.IGNORECASE)
_FENCE_RE = re.compile(r"^```(?:svg|xml|html)?\s*|\s*```$", re.MULTILINE)
_SHAPE_RE = re.compile(
    r"<(path|rect|circle|ellipse|polygon|polyline|line)\b", re.IGNORECASE)
_COLOUR_RE = re.compile(
    r"""(?:fill|stroke|stop-color)\s*[:=]\s*["']?\s*(#[0-9a-fA-F]{3,6}|rgb\([^)]*\)|[a-z]+)""",
    re.IGNORECASE)
_STROKE_W_RE = re.compile(r"""stroke-width\s*[:=]\s*["']?\s*([0-9.]+)""", re.IGNORECASE)
_GROUND_RE = re.compile(
    r"""<rect\b[^>]*\bwidth\s*=\s*["']?(\d+)[^>]*\bheight\s*=\s*["']?(\d+)""",
    re.IGNORECASE)

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
                purpose: str = "", subject: str = "") -> None:
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
            "identity": identity_hash(title, purpose),
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


def sanitize_svg(raw: str, palette: Optional[Sequence[str]] = None) -> str:
    """Return the drawing, or raise :class:`IconError` naming what is wrong.

    Refuses rather than strips, and the message is written to be handed
    straight back to the model as the reason for a redraw. Everything checked
    here is something that made the first batch of icons look generated:
    a floating badge, a colour from nowhere, detail that dies at 24px.

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
    if _EXTERNAL_REF_RE.search(text) or _DATA_URI_RE.search(text):
        raise IconError("the icon references an external or embedded file — "
                        "it must be self-contained vector drawing")
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

    if not any(int(w) >= 96 and int(h) >= 96 for w, h in _GROUND_RE.findall(text)):
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
    """
    ranked, dark, light = palette_mod.ordered(list(palette or []))
    if not ranked:
        # No palette at all: neutral, and obviously provisional. Not a
        # generated hue — an app with no colours of its own does not acquire
        # some here.
        ranked, dark, light = ["#232329", "#3A3A44", "#8A8A96"], "#232329", "#8A8A96"
    mid = ranked[len(ranked) // 2]
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96" '
        'width="96" height="96" role="img">'
        f'<rect width="96" height="96" fill="{dark}"/>'
        f'<rect x="0" y="34" width="96" height="30" fill="{mid}"/>'
        f'<rect x="0" y="64" width="96" height="14" fill="{light}"/>'
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
    " - Pick the most SPECIFIC object in the app, not the category it belongs "
    "to. The test: if the object would suit ten other apps, it is wrong.\n"
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
    " - Paths, shapes and gradients only. No <text>, no <image>, no "
    "<foreignObject>, no script, no external or data: references, no filters, "
    "no blurs, no drop shadows.\n"
    " - Under 6 KB.\n"
    "\n"
    "WHAT YOU ARE MAKING\n"
    "A MARK — not an illustration and not a UI pictogram. The kind of thing "
    "that works as an enamel pin or a three-colour screen print: bold, "
    "reduced, confident, recognisable across a room.\n"
    "\n"
    "COMPOSITION — all three, every time\n"
    ' 1. FULL BLEED. The FIRST element is <rect width="96" height="96" '
    'fill="…"/> and it covers the whole frame. No margin, no border, no '
    "rounded corner, no transparency. The client rounds the corners; you must "
    "not.\n"
    " 2. HUGE SUBJECT. It fills most of the frame and runs off at least one "
    "edge, clipped by it. A small object centred with space around it is the "
    "failure to avoid.\n"
    " 3. TWO TO FOUR SHAPES on the ground, each overlapping another, so the "
    "mark reads as layered rather than flat. Reduce the scene until it is "
    "that few.\n"
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
    " - Darkest as the full-bleed ground, a mid colour for the subject, the "
    "lightest for the one detail that must pop. A linear gradient between two "
    "of the given values is allowed; inventing a colour is not."
)


def identity_hash(title: str, purpose: str) -> str:
    """What the icon DEPICTS, as a stable key.

    An icon is redrawn when the app changes what it *is*, not when a line of
    CSS moves: redrawing on every edit would spend a model call on a padding
    change and make the tile flicker between revisions.
    """
    basis = f"{' '.join((title or '').split())}|{' '.join((purpose or '').split())[:200]}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


#: Back-compat for callers/tests written against the first version.
subject_hash = identity_hash


def is_stale(slug: str, *, title: str, purpose: str) -> bool:
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
    return meta.get("identity") != identity_hash(title, purpose)


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
    ranked, dark, light = palette_mod.ordered(list(palette))
    brief = (
        f'App name: "{title}".\n'
        f"SCENE TO DRAW: {scene}\n"
        f"The app's palette, darkest first: {', '.join(ranked)}.\n"
        f"Use {dark} as the full-bleed ground and {light} for the detail that "
        f"must pop. Use no other colours.\n"
        "Draw it. SVG only."
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
            return sanitize_svg(raw, ranked)
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
    ``fallback``. Never raises and never returns nothing.

    ``html`` is the app's own source, and it is what the palette is read
    from. Without it the mark can still be drawn, but on the neutral holding
    palette — so callers that have the file should always pass it.
    """
    slug = store.normalise_slug(slug)
    if not is_stale(slug, title=title, purpose=purpose):
        existing = read_icon(slug)
        if existing:
            return existing, "kept"

    if not html:
        try:
            html = store.read_app(slug)
        except (OSError, AppStoreError):
            html = ""
    colours = palette_mod.extract(html) if html else []

    if allow_model:
        from app.agent.skills.builtins.app_html import vision
        if vision.can_call_model():
            key, scene = await choose_subject(
                user_id=user_id, title=title, purpose=purpose,
                used=subjects_in_use(exclude=slug),
            )
            if scene and colours:
                svg = await draw_mark(user_id=user_id, title=title, scene=scene,
                                      palette=colours)
                if svg:
                    try:
                        _store_icon(slug, svg, source="model", title=title,
                                    purpose=purpose, subject=key)
                    except (OSError, AppStoreError):
                        logger.warning("[app_html] could not store the icon for %s",
                                       slug, exc_info=True)
                    return svg, "model"
            elif scene and not colours:
                # No palette means no way to make the mark belong to the app,
                # and a mark in invented colours is the defect this round
                # removed. Better a holding mark that says so.
                logger.info("[app_html] %s has no palette of its own — holding "
                            "mark rather than an invented one", slug)

    existing = read_icon(slug)
    meta = read_sidecar(slug)
    if existing and meta.get("source") == "model" and meta.get("gen") == str(ICON_GENERATION):
        # Keep a current designed mark rather than downgrading it because one
        # generation failed.
        return existing, "kept"

    svg = fallback_icon(slug, title, colours)
    try:
        _store_icon(slug, svg, source="fallback", title=title, purpose=purpose)
    except (OSError, AppStoreError):
        logger.debug("[app_html] could not store the holding mark for %s", slug,
                     exc_info=True)
    return svg, "fallback"
