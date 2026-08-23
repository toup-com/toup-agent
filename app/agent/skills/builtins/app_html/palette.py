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

Round 25 adds :func:`roles`, because reading the palette correctly was only
half the job — the other half is deciding which of those colours the SUBJECT
is drawn in, and until now that was luminance alone. See its docstring: for
the design skill's own dark example that handed the glyph to ``--muted`` and
left ``--accent`` with no role at all, so a dark app got a grey-on-grey mark.
"""

from __future__ import annotations

import re
from typing import Dict, List, NamedTuple, Tuple

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

#: The floor a glyph has to clear against its ground to be picked FOR being
#: chromatic. WCAG 2.1 SC 1.4.11 (non-text contrast) is 3:1 for a graphical
#: object that carries meaning, and an app icon at 24px is exactly that. A
#: chromatic colour that fails it is not dropped — it is demoted to the
#: detail role, where it sits on the glyph rather than on the ground.
MIN_GLYPH_CONTRAST = 3.0

#: The detail is usually drawn ON the subject, so it does not need the full
#: 3:1 against the ground — but it must not BE the ground. Two is the point
#: where a second colour is still legible as a separate shape.
MIN_DETAIL_CONTRAST = 2.0


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
    prompt can say "darkest as the ground" and mean something.

    This is the un-linearised weighted sum, which is monotonic in brightness
    and therefore fine for ORDERING. It is deliberately not what
    :func:`contrast` uses: a contrast RATIO computed without the sRGB
    transfer curve is wrong by a wide margin at the dark end — burnt orange
    on dark green measures 2.7 by this function and 4.6 by the real one, and
    3.0 is a threshold this module makes decisions on.
    """
    value = normalise(hex_colour)[1:]
    r, g, b = (int(value[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _channels(hex_colour: str) -> Tuple[float, float, float]:
    value = normalise(hex_colour)[1:]
    return tuple(int(value[i:i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def relative_luminance(hex_colour: str) -> float:
    """WCAG 2.1 relative luminance — the sRGB channels linearised first."""
    out = 0.0
    for channel, weight in zip(_channels(hex_colour), (0.2126, 0.7152, 0.0722)):
        linear = (channel / 12.92 if channel <= 0.03928
                  else ((channel + 0.055) / 1.055) ** 2.4)
        out += weight * linear
    return out


def contrast(a: str, b: str) -> float:
    """WCAG contrast ratio between two colours, 1.0 (identical) → 21.0."""
    la, lb = relative_luminance(a), relative_luminance(b)
    if lb > la:
        la, lb = lb, la
    return (la + 0.05) / (lb + 0.05)


def chroma(hex_colour: str) -> float:
    """How much colour a colour has, 0 (grey) → 1 (fully saturated).

    ``max - min`` over the RGB channels: HSV saturation without the division
    by value, so a dark saturated navy does not outrank a bright orange the
    way plain HSL saturation would. What this has to separate is "the app's
    one accent" from "the app's greys", and that is what it separates.
    """
    r, g, b = _channels(hex_colour)
    return max(r, g, b) - min(r, g, b)


class Roles(NamedTuple):
    """Which palette colour plays which part in the mark."""

    ranked: List[str]   #: dark → light, the whole palette
    ground: str         #: the full-bleed rect
    glyph: str          #: the subject itself
    detail: str         #: the one thing that has to pop


def roles(palette: List[str]) -> Roles:
    """Assign the mark's three parts, by chroma and contrast — not luminance.

    Round 25. Until now the roles came out of :func:`ordered`, which sorts by
    luminance and hands back ``ranked[0]`` and ``ranked[-1]``; the subject got
    "a mid colour" and which mid colour that was depended entirely on where
    the app's accent happened to sort. Worked through on the design skill's
    own two examples (``DESIGN_SKILL.md`` §1d):

    * light — ``#FFF8F0 #FFFFFF #1A1410 #7A6A5D #F0552B`` — the middle by
      luminance IS ``--accent`` (#F0552B). Right, by luck.
    * dark — ``#0E1424 #18203A #E8EAF2 #8A93AC #E3A857`` — the middle by
      luminance is ``--muted`` (#8A93AC), and ``--accent`` (#E3A857) gets no
      role at all. The app's one chromatic colour is absent from its own
      icon and the mark is grey on near-black.

    So the ground stays what round 20 decided — the darkest colour, which is
    what makes the library read as one shelf — and the GLYPH becomes the most
    chromatic colour that clears :data:`MIN_GLYPH_CONTRAST` against it. Pure,
    total, and no I/O: every branch is unit-testable.
    """
    ranked = sorted(palette, key=luminance)
    if not ranked:
        return Roles([], "", "", "")
    ground = ranked[0]
    rest = [c for c in ranked[1:] if normalise(c) != normalise(ground)]
    if not rest:
        return Roles(ranked, ground, ground, ground)

    # The glyph: the most chromatic colour that is legible on the ground. If
    # the accent is too close to the ground to read at 24px, readability wins
    # and the accent is demoted to the detail below — a mark nobody can make
    # out is worse than a mark that is not in the brand colour.
    legible = [c for c in rest if contrast(c, ground) >= MIN_GLYPH_CONTRAST]
    if legible:
        glyph = max(legible, key=lambda c: (chroma(c), contrast(c, ground)))
    else:
        glyph = max(rest, key=lambda c: contrast(c, ground))

    remaining = [c for c in rest if normalise(c) != normalise(glyph)]
    if not remaining:
        return Roles(ranked, ground, glyph, glyph)
    # The detail has to be visible against the subject it sits on AND not be
    # the ground wearing a different name.
    off_ground = [c for c in remaining
                  if contrast(c, ground) >= MIN_DETAIL_CONTRAST]
    detail = max(off_ground or remaining, key=lambda c: contrast(c, glyph))
    return Roles(ranked, ground, glyph, detail)


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

    Kept as it is for its callers and for the ordering itself, which is still
    how the palette is PRESENTED to the model. Role assignment moved to
    :func:`roles` in round 25 — see its docstring for why luminance alone was
    the wrong basis for deciding what the subject is painted in.
    """
    if not palette:
        return [], "", ""
    ranked = sorted(palette, key=luminance)
    return ranked, ranked[0], ranked[-1]
