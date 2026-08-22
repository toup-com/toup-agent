"""Every app gets a mark that says what it is.

Round 20, item 4. An app card carried a generic glyph, so a library of
twenty apps was twenty identical tiles distinguishable only by reading them.
The fix is a real icon per app — a snake for the snake game, a timer for the
pomodoro — designed at build time and stored beside the artifact.

**SVG, not a raster.** It is a few hundred bytes instead of a few hundred
kilobytes, it is sharp at every size a card, a tab and a home screen ask for,
it needs no image API, no credit charge and no second network hop, and — the
part that decides it — it is *text*, so the model that just wrote the app can
draw it, and this module can read it and refuse the parts that are not
drawing.

**There is always an icon.** The designed one needs a model; a container that
cannot reach one still gets :func:`fallback_icon`, a deterministic mark
derived from the slug, so no card ever renders a broken image. The fallback
records itself as a fallback (`source: "fallback"` in the sidecar) so the
next run that CAN reach a model upgrades it — an icon that silently stays
generic forever is the failure this item exists to fix, and it would look
exactly like success.

**What comes back from the model is not trusted.** An SVG is a document that
can carry script, external references and event handlers; this one is
rendered by the shell, on the shell's own origin, in an ``<img>`` and a card.
:func:`sanitize_svg` refuses every one of those rather than stripping them —
a mark that has to be edited to be safe is a mark the model should draw
again.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from typing import Dict, Optional, Tuple

from app.agent.skills.builtins.app_html import store
from app.agent.skills.builtins.app_html.store import AppStoreError

logger = logging.getLogger(__name__)

#: Dot-directory: invisible to the library scanner by name, by depth and by
#: suffix, the same three ways the brief is (see `appskill`).
ICON_DIR = ".icons"
ICON_SUFFIX = ".svg"
SIDECAR_SUFFIX = ".json"

#: Model that draws the mark. Pinned for the same reason `vision.REVIEW_MODEL`
#: is: `model=None` bills a tenant's chat model for a background job.
LOGO_MODEL = "gpt-4o-mini"
LOGO_TIMEOUT_S = 30

#: A mark is a few hundred bytes of path data. This ceiling is what separates
#: "an icon" from "someone pasted a traced photograph".
MAX_ICON_BYTES = 24 * 1024
MIN_ICON_BYTES = 80

#: Everything an SVG can do that is not drawing.
_FORBIDDEN_TAGS = re.compile(
    r"<\s*(script|foreignObject|iframe|embed|object|audio|video|animate|set|"
    r"handler|use\b[^>]*\bxlink:href\s*=\s*[\"']https?:)",
    re.IGNORECASE,
)
_EVENT_ATTR_RE = re.compile(r"""\son[a-z]+\s*=\s*["']""", re.IGNORECASE)
_EXTERNAL_REF_RE = re.compile(
    r"""(?:href|xlink:href|src)\s*=\s*["']\s*(?:https?:)?//""", re.IGNORECASE
)
_DATA_URI_RE = re.compile(r"""(?:href|xlink:href|src)\s*=\s*["']\s*data:""", re.IGNORECASE)
_SVG_OPEN_RE = re.compile(r"<svg\b[^>]*>", re.IGNORECASE)
_VIEWBOX_RE = re.compile(r"\bviewBox\s*=\s*[\"'][^\"']+[\"']", re.IGNORECASE)
_FENCE_RE = re.compile(r"^```(?:svg|xml|html)?\s*|\s*```$", re.MULTILINE)


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


def _store_icon(slug: str, svg: str, *, source: str, title: str, purpose: str) -> None:
    path = icon_path(slug)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    store.repair_permissions(os.path.dirname(path))
    store._atomic_write(path, svg.encode("utf-8"), prefix=f".icon-{slug}-")
    store._atomic_write(
        sidecar_path(slug),
        json.dumps({
            "slug": slug,
            "source": source,
            "title": title or slug,
            "subject": subject_hash(title, purpose),
        }, sort_keys=True).encode("utf-8"),
        prefix=f".iconmeta-{slug}-",
    )


# ── Validation ────────────────────────────────────────────────────────

def sanitize_svg(raw: str) -> str:
    """Return the drawing, or raise :class:`IconError` naming what is wrong.

    Refuses rather than strips. The message goes back to the model, which can
    draw it again — and a stripped SVG is a drawing nobody has looked at,
    which is the state this whole round is about not shipping.
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
            f"limit — an icon is paths, not a traced photograph"
        )
    if _FORBIDDEN_TAGS.search(text):
        raise IconError("the icon contains script or embedded content — paths, "
                        "shapes, text and gradients only")
    if _EVENT_ATTR_RE.search(text):
        raise IconError("the icon has an on… event attribute — an icon does not run")
    if _EXTERNAL_REF_RE.search(text) or _DATA_URI_RE.search(text):
        raise IconError("the icon references an external or embedded file — "
                        "it must be self-contained vector drawing")
    if not _VIEWBOX_RE.search(_SVG_OPEN_RE.search(text).group(0) if _SVG_OPEN_RE.search(text) else ""):
        raise IconError("the <svg> has no viewBox — without one it cannot scale")
    return text


# ── The always-available mark ─────────────────────────────────────────

_HUES = (14, 32, 48, 96, 152, 178, 199, 217, 245, 268, 292, 330)


def initials(title: str, slug: str) -> str:
    """One or two letters, from the name a person reads."""
    words = [w for w in re.split(r"[^A-Za-z0-9]+", title or slug or "") if w]
    if not words:
        return "?"
    if len(words) == 1:
        return words[0][:2].upper()
    return (words[0][0] + words[1][0]).upper()


def fallback_icon(slug: str, title: str = "") -> str:
    """A mark derived from the slug — no model, no network, always the same.

    Deterministic so an app's tile does not change colour between two
    containers, and so a test can assert on it. It is a monogram, not a
    depiction: it exists so a card is never empty and never broken, and it
    marks itself in the sidecar so a real icon replaces it later.
    """
    digest = hashlib.sha256((slug or "app").encode("utf-8")).digest()
    hue = _HUES[digest[0] % len(_HUES)]
    hue2 = (hue + 28) % 360
    text = initials(title, slug)
    size = 34 if len(text) > 1 else 40
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96" '
        'width="96" height="96" role="img">'
        f'<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0" stop-color="hsl({hue} 72% 58%)"/>'
        f'<stop offset="1" stop-color="hsl({hue2} 68% 42%)"/>'
        f'</linearGradient></defs>'
        '<rect width="96" height="96" rx="22" fill="url(#g)"/>'
        f'<text x="48" y="48" text-anchor="middle" dominant-baseline="central" '
        f'font-family="ui-sans-serif, system-ui, -apple-system, Segoe UI, '
        f'Helvetica, Arial, sans-serif" font-size="{size}" font-weight="700" '
        f'fill="#fff" letter-spacing="-1">{text}</text>'
        '</svg>'
    )


# ── The designed mark ─────────────────────────────────────────────────

_SYSTEM = (
    "You draw app icons as SVG. One icon, one subject, no text.\n"
    "\n"
    "Rules:\n"
    " - Output ONLY the SVG. No prose, no markdown fences, no explanation.\n"
    ' - Exactly: <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96" '
    'width="96" height="96"> … </svg>\n'
    " - DEPICT THE SUBJECT. A snake game gets a snake. A pomodoro timer gets "
    "a timer. A budget tracker gets a coin, a wallet or a chart. Never an "
    "abstract blob, never a generic document, never a letter or monogram.\n"
    " - No <text>. An icon that has to be read is not an icon.\n"
    " - Draw on a filled rounded-square field: "
    '<rect width="96" height="96" rx="22" .../> as the first element, then '
    "the subject on top of it, with a clear silhouette and generous margins "
    "(keep the subject inside roughly x/y 18–78).\n"
    " - Two or three colours, high contrast against the field, so it reads at "
    "24px. Flat shapes or a single linear gradient; no filters, no blurs, no "
    "shadows.\n"
    " - Paths, shapes and gradients only. No script, no event attributes, no "
    "external or data: references, no <image>, no <foreignObject>.\n"
    " - Under 6 KB."
)


def subject_hash(title: str, purpose: str) -> str:
    """What the icon DEPICTS, as a stable key.

    An icon is regenerated when the app changes what it *is*, not every time
    a line of CSS moves — redrawing the mark on every edit would make a
    tile that flickers between revisions and would spend a model call on a
    padding change. The name plus the first line of the brief's narrative is
    the app's identity in words; when that moves, the drawing is stale.
    """
    basis = f"{' '.join((title or '').split())}|{' '.join((purpose or '').split())[:200]}"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:16]


def is_stale(slug: str, *, title: str, purpose: str) -> bool:
    """Does this app need a (re)drawn icon?"""
    if read_icon(slug) is None:
        return True
    meta = read_sidecar(slug)
    if meta.get("source") != "model":
        # A fallback monogram is a placeholder by definition; upgrade it the
        # first time a model is reachable.
        return True
    return meta.get("subject") != subject_hash(title, purpose)


async def draw_icon(
    *, user_id: str, title: str, purpose: str, slug: str,
) -> Optional[str]:
    """Ask the model for the mark. None if it could not be drawn."""
    from app.agent.skills.builtins.app_html import vision
    if not vision.can_call_model():
        # Cheap and first. Without it, every publish on a container with no
        # credential spends the full `LOGO_TIMEOUT_S` discovering the same
        # thing, and the user waits half a minute for a monogram.
        return None
    brief = [f'App name: "{title or slug}".']
    if purpose:
        brief.append(f"What it is: {' '.join(purpose.split())[:400]}")
    brief.append(
        "Draw its icon: one recognisable object that says what this app is. "
        "Output the SVG only."
    )
    try:
        from app.services.internal_llm import call_system_llm
        raw = await asyncio.wait_for(
            call_system_llm(
                user_id=user_id or "",
                operation_type="system.app_html.icon",
                model=LOGO_MODEL,
                max_tokens=1600,
                system=_SYSTEM,
                messages=[{"role": "user", "content": "\n".join(brief)}],
                timeout=LOGO_TIMEOUT_S,
            ),
            timeout=LOGO_TIMEOUT_S,
        )
    except asyncio.TimeoutError:
        logger.warning("[app_html] icon generation timed out for %s", slug)
        return None
    except Exception:  # noqa: BLE001 - a missing model must not fail a build
        logger.warning("[app_html] icon generation could not run for %s", slug,
                       exc_info=True)
        return None
    if not raw:
        return None
    try:
        return sanitize_svg(raw)
    except IconError as exc:
        logger.warning("[app_html] model's icon for %s refused: %s", slug, exc)
        return None


async def ensure_icon(
    slug: str, *, title: str, purpose: str = "", user_id: str = "",
    allow_model: bool = True,
) -> Tuple[str, str]:
    """The app's icon, drawing it if it is missing or stale.

    Returns ``(svg, source)`` where source is ``kept`` | ``model`` |
    ``fallback``. Never raises and never returns nothing: an app always has a
    mark, even on a container that cannot reach a model.
    """
    slug = store.normalise_slug(slug)
    if not is_stale(slug, title=title, purpose=purpose):
        existing = read_icon(slug)
        if existing:
            return existing, "kept"

    if allow_model:
        svg = await draw_icon(user_id=user_id, title=title, purpose=purpose, slug=slug)
        if svg:
            try:
                _store_icon(slug, svg, source="model", title=title, purpose=purpose)
                return svg, "model"
            except (OSError, AppStoreError):
                logger.warning("[app_html] could not store the icon for %s", slug,
                               exc_info=True)
                return svg, "model"

    # Keep an existing designed mark rather than downgrading it to a monogram
    # because one generation failed.
    existing = read_icon(slug)
    if existing and read_sidecar(slug).get("source") == "model":
        return existing, "kept"

    svg = fallback_icon(slug, title)
    try:
        _store_icon(slug, svg, source="fallback", title=title, purpose=purpose)
    except (OSError, AppStoreError):
        logger.debug("[app_html] could not store the fallback icon for %s", slug,
                     exc_info=True)
    return svg, "fallback"
