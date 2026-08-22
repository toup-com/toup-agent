"""The colours an app actually uses, read out of the app.

Round 20, logo correction. The first icons picked their colours from a hash
of the slug — a twelve-hue ramp with no relationship to the app at all. So a
mole game whose whole screen is dark green and burnt orange got a tile in
whatever hue its slug hashed to, and the library looked like a bag of
sweets rather than a shelf of that person's things.

An app already HAS a palette: the model chose one when it wrote the CSS, and
the design skill asks for it as custom properties on ``:root``. That is the
palette the icon has to belong to, so it is read from the file rather than
invented.

Two passes, in order of how much the author meant them:

1. **CSS custom properties.** ``--bg``, ``--ink``, ``--accent`` — these are
   design tokens, declared deliberately, in the order the author thought of
   them. If an app has them, they ARE the palette.
2. **Literal colours by frequency**, for an app that hard-coded everything.
   A colour used twelve times is a theme; one used once is a detail.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

#: Enough to build a mark from (ground, subject, one accent, one highlight)
#: and few enough that the model cannot spread it thin.
MAX_PALETTE = 5

#: Below this many distinct colours the app has no palette worth speaking of
#: and the caller should say so rather than pass one colour along.
MIN_PALETTE = 2

_VAR_RE = re.compile(r"--([a-z0-9-]{1,40})\s*:\s*(#[0-9a-fA-F]{3,8})\s*[;}]")
_HEX_RE = re.compile(r"#([0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b")
_STYLE_RE = re.compile(r"<style\b[^>]*>(.*?)</style\s*>", re.IGNORECASE | re.DOTALL)

#: Greys that are almost certainly not the app's identity: pure black and
#: pure white turn up in every reset and every shadow. Kept only when the app
#: has nothing else, because an app really can be black-and-white.
_NEUTRAL = {"#000000", "#ffffff"}


def normalise(hex_colour: str) -> str:
    """``#ABC`` → ``#aabbcc``; anything longer is truncated to its RGB."""
    value = hex_colour.strip().lower()
    if not value.startswith("#"):
        value = "#" + value
    body = value[1:]
    if len(body) == 3:
        body = "".join(c * 2 for c in body)
    return "#" + body[:6]


def luminance(hex_colour: str) -> float:
    """Relative luminance, 0–1. Used to order the palette dark → light so the
    prompt can say "darkest as the ground" and mean something."""
    value = normalise(hex_colour)[1:]
    r, g, b = (int(value[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def extract(html: str) -> List[str]:
    """The app's palette, most deliberate first, at most :data:`MAX_PALETTE`.

    Returns ``[]`` when the app has no usable colours of its own — which the
    caller must treat as "no palette", never as "use a default one". An
    invented palette is exactly the defect this module exists to remove.
    """
    styles = "\n".join(_STYLE_RE.findall(html or ""))
    out: List[str] = []

    for _name, value in _VAR_RE.findall(styles):
        colour = normalise(value)
        if colour not in out:
            out.append(colour)
        if len(out) >= MAX_PALETTE:
            return out

    if len(out) >= MIN_PALETTE:
        return out

    counts: Dict[str, int] = {}
    for match in _HEX_RE.findall(styles or html or ""):
        colour = normalise("#" + match)
        counts[colour] = counts.get(colour, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    for colour, _n in ranked:
        if colour in out:
            continue
        if colour in _NEUTRAL and len(counts) > len(_NEUTRAL):
            continue
        out.append(colour)
        if len(out) >= MAX_PALETTE:
            break
    return out if len(out) >= MIN_PALETTE else []


def ordered(palette: List[str]) -> Tuple[List[str], str, str]:
    """``(dark→light, darkest, lightest)``.

    The prompt assigns roles by value — darkest is the ground, lightest is
    the detail that has to pop — so the ordering has to be computed, not
    assumed from declaration order.
    """
    if not palette:
        return [], "", ""
    ranked = sorted(palette, key=luminance)
    return ranked, ranked[0], ranked[-1]
