"""Build the round-25 icon contact sheet, and check every exemplar first.

The repo's method for an aesthetic requirement (see `feedback_generated_art
_needs_a_contact_sheet`): generate a set, render it at three sizes with the
source palette beside it, and LOOK. Round 20 found its defects that way and
would not have found them any other way.

There are no model credentials on this machine, so the marks here are drawn
BY HAND to the new spec rather than generated. That means this sheet proves
two things and not a third:

* the round-25 spec is drawable — six subjects, six palettes, all inside the
  safe area, all through `logo.sanitize_svg` with their own palette;
* what the spec LOOKS like at 96, 48 and 24 px on both backdrops.

It does not prove what the model draws when asked. That needs credentials.

    RUN_MODE=platform PYTHONPATH=. python scripts/icon_contact_sheet.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.agent.skills.builtins.app_html import logo, palette  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
EXEMPLAR_DIR = os.path.join(_HERE, "icon_exemplars")
OUT = os.path.join(_HERE,
                   "icon_contact_sheet.html")

# Two of these palettes are the design skill's own worked examples
# (DESIGN_SKILL.md §1d) so the sheet is not judging colours invented for it.
EXEMPLARS = [
    dict(
        name="Sleep wind-down",
        subject="crescent moon over a low hill",
        note="dark app; the crescent is a knockout in the ground colour",
        palette=["#0E1424", "#18203A", "#E8EAF2", "#8A93AC", "#E3A857"],
        body='<circle cx="46" cy="46" r="30" fill="#E3A857"/>'
             '<circle cx="64" cy="34" r="26" fill="#0E1424"/>'
             '<path d="M18 78 C30 62 44 62 56 74 C62 79 70 80 78 78 L78 82 '
             'L18 82 Z" fill="#8A93AC"/>',
    ),
    dict(
        name="Gym log",
        subject="dumbbell, seen end-on and from the side",
        note="light app; ground stays the darkest colour, glyph is --accent",
        palette=["#FFF8F0", "#FFFFFF", "#1A1410", "#7A6A5D", "#F0552B"],
        body='<rect x="14" y="34" width="16" height="28" rx="4" fill="#F0552B"/>'
             '<rect x="66" y="34" width="16" height="28" rx="4" fill="#F0552B"/>'
             '<rect x="28" y="42" width="40" height="12" fill="#FFFFFF"/>',
    ),
    dict(
        name="Weeknight recipes",
        subject="a covered pot with one handle",
        note="light app, warm palette",
        palette=["#FBF3E6", "#FFFFFF", "#2A1D14", "#8A7460", "#C1452B"],
        body='<path d="M20 44 L76 44 L70 78 L26 78 Z" fill="#C1452B"/>'
             '<rect x="16" y="34" width="64" height="10" rx="5" fill="#FFFFFF"/>'
             '<rect x="42" y="24" width="12" height="12" rx="6" fill="#8A7460"/>',
    ),
    dict(
        name="Whack a Mole",
        subject="mole snout rising out of its hole",
        note="the round-20 test palette, so the two directions are comparable",
        palette=["#1E2E1C", "#3E2C1E", "#E2703A", "#F4F1E6"],
        # First draft of this one was REFUSED — bbox centre 48,66, because a
        # subject "rising out of a hole" anchors itself to the bottom of the
        # frame without noticing. Redrawn to fill the safe box instead of
        # sitting in the bottom third of it. The validator caught a real
        # composition fault in a mark a human had just drawn on purpose.
        body='<path d="M22 74 C22 2 74 2 74 74 Z" fill="#E2703A"/>'
             '<ellipse cx="48" cy="74" rx="33" ry="8" fill="#3E2C1E"/>'
             '<circle cx="38" cy="46" r="5" fill="#F4F1E6"/>'
             '<circle cx="58" cy="46" r="5" fill="#F4F1E6"/>',
    ),
    dict(
        name="Tide table",
        subject="a wave curling over itself",
        note="two-colour palette plus a near-white",
        palette=["#0B2B2E", "#2FA39B", "#F2F7F7"],
        body='<path d="M16 68 C16 30 64 22 76 44 C82 56 70 66 60 60 C54 56 '
             '56 46 64 46 L64 34 C40 34 30 50 34 68 Z" fill="#2FA39B"/>'
             '<circle cx="66" cy="52" r="7" fill="#F2F7F7"/>',
    ),
    dict(
        name="Nokia Snake",
        subject="the snake's body turning a corner toward the pellet",
        note="the fixture every test in test_app_logo.py is checked against",
        palette=["#2F6B3A", "#F7F4EC", "#C4703A"],
        body='<path d="M20 62 Q34 40 48 54 Q62 68 76 46" stroke="#F7F4EC" '
             'stroke-width="10" fill="none" stroke-linecap="round"/>'
             '<circle cx="72" cy="42" r="7" fill="#C4703A"/>',
    ),
]


def build(item: dict) -> str:
    parts = palette.roles(item["palette"])
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96" '
        'width="96" height="96" role="img">'
        f'<rect width="96" height="96" fill="{parts.ground}"/>'
        f'{item["body"]}</svg>'
    )


def main() -> int:
    rows = []
    bad = 0
    for item in EXEMPLARS:
        svg = build(item)
        parts = palette.roles(item["palette"])
        try:
            logo.sanitize_svg(svg, item["palette"])
            verdict = "passes the validator"
        except logo.IconError as exc:
            verdict = f"REFUSED: {exc}"
            bad += 1
        box = logo.measure_glyph(svg)
        measured = (
            f"{box[0]:.0f},{box[1]:.0f} → {box[2]:.0f},{box[3]:.0f} · "
            f"{box[2] - box[0]:.0f}×{box[3] - box[1]:.0f} · centre "
            f"{(box[0] + box[2]) / 2:.0f},{(box[1] + box[3]) / 2:.0f}"
            if box else "unmeasurable")
        swatches = "".join(
            f'<i style="background:{c}" title="{c}"></i>' for c in item["palette"])
        roles_line = (f"ground {parts.ground} · glyph {parts.glyph} · "
                      f"detail {parts.detail}")
        sizes = "".join(
            f'<div class="s"><div class="tile" style="width:{n}px;height:{n}px">'
            f'{svg}</div><b>{n}px</b></div>' for n in (96, 48, 24))
        rows.append(
            f'<tr><td class="meta"><h3>{item["name"]}</h3>'
            f'<p class="sub">{item["subject"]}</p>'
            f'<p class="note">{item["note"]}</p>'
            f'<p class="sw">{swatches}</p>'
            f'<p class="mono">{roles_line}</p>'
            f'<p class="mono">glyph box {measured}</p>'
            f'<p class="mono v">{verdict}</p></td>'
            f'<td class="light"><div class="strip">{sizes}</div></td>'
            f'<td class="dark"><div class="strip">{sizes}</div></td></tr>')

    # The thresholds themselves, at the size that decides them. Numbers
    # chosen by arithmetic ("18 units is 4.5px of a 24px tile") are a guess
    # until they are looked at next to the alternatives.
    probe_pal = ["#1E2E1C", "#E2703A", "#F4F1E6"]
    probes = []
    for label, long, short in [("40×40 (the long-axis floor)", 40, 40),
                               ("54×54", 54, 54),
                               ("68×68 (fills the safe box)", 68, 68),
                               ("68×18 (the old short-axis floor)", 68, 18),
                               ("68×24 (the short-axis floor now)", 68, 24),
                               ("68×32", 68, 32)]:
        x, y = 48 - long / 2, 48 - short / 2
        svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96" '
               'width="96" height="96"><rect width="96" height="96" '
               'fill="#1E2E1C"/>'
               f'<rect x="{x}" y="{y}" width="{long}" height="{short}" rx="4" '
               'fill="#E2703A"/>'
               f'<circle cx="48" cy="48" r="{min(long, short) / 4:.1f}" '
               'fill="#F4F1E6"/></svg>')
        try:
            logo.sanitize_svg(svg, probe_pal)
            state = "accepted"
        except logo.IconError:
            state = "refused"
        cells = "".join(
            f'<div class="s"><div class="tile" style="width:{n}px;height:{n}px">'
            f'{svg}</div><b>{n}px</b></div>' for n in (96, 48, 24))
        probes.append(f'<tr><td class="meta"><p class="mono">{label}</p>'
                      f'<p class="mono v">{state}</p></td>'
                      f'<td class="light"><div class="strip">{cells}</div></td>'
                      f'<td class="dark"><div class="strip">{cells}</div></td>'
                      f'</tr>')

    html = f"""<!doctype html><meta charset="utf-8">
<title>Round 25 icon contact sheet</title>
<style>
 body{{font:14px/1.5 ui-sans-serif,system-ui,sans-serif;margin:24px;
      background:#f6f6f7;color:#111}}
 h1{{font-size:20px;margin:0 0 4px}}
 p.lede{{margin:0 0 20px;color:#555;max-width:70ch}}
 table{{border-collapse:collapse;width:100%}}
 td{{border-top:1px solid #ddd;padding:14px;vertical-align:middle}}
 td.meta{{width:300px}}
 td.light{{background:#ffffff}}
 td.dark{{background:#101014}}
 h3{{margin:0;font-size:15px}}
 .sub{{margin:2px 0;color:#444}}
 .note{{margin:2px 0;color:#777;font-size:12px}}
 .mono{{font:11px/1.4 ui-monospace,Menlo,monospace;color:#666;margin:2px 0}}
 .v{{color:#0a6}}
 .sw i{{display:inline-block;width:22px;height:22px;border-radius:3px;
        margin-right:4px;border:1px solid rgba(0,0,0,.15)}}
 .strip{{display:flex;align-items:flex-end;gap:20px}}
 .s{{text-align:center}}
 .s b{{display:block;font:10px ui-monospace,monospace;color:#999;margin-top:6px}}
 td.dark .s b{{color:#777}}
 .tile{{overflow:hidden;border-radius:22.37%}}
 .tile svg{{width:100%;height:100%;display:block}}
</style>
<h1>Round 25 — icon contact sheet</h1>
<p class="lede">Six marks drawn by hand to the new direction: one glyph,
centred, inside the safe area from 14 to 82 of the 96 frame, over a full-bleed
ground. Each is rendered at 96, 48 and 24 px, masked the way the client masks
it, on a light and a dark backdrop, with its source palette beside it. The
24px column is the one that decides.</p>
<table>{''.join(rows)}</table>
<h1 style="margin-top:34px">The size thresholds, at 24px</h1>
<p class="lede">A plain block at each threshold, so
<code>ICON_MIN_GLYPH_LONG</code> and <code>ICON_MIN_GLYPH_SHORT</code> are
chosen by looking rather than by arithmetic. Same palette throughout.</p>
<table>{''.join(probes)}</table>
"""
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(html)
    os.makedirs(EXEMPLAR_DIR, exist_ok=True)
    for item in EXEMPLARS:
        slug = item["name"].lower().replace(" ", "-")
        with open(os.path.join(EXEMPLAR_DIR, slug + ".svg"), "w",
                  encoding="utf-8") as fh:
            fh.write(build(item))
    print(f"wrote {OUT}")
    print(f"wrote {len(EXEMPLARS)} exemplars to {EXEMPLAR_DIR}")
    for item in EXEMPLARS:
        svg = build(item)
        try:
            logo.sanitize_svg(svg, item["palette"])
            print(f"  ok       {item['name']}  {logo.measure_glyph(svg)}")
        except logo.IconError as exc:
            print(f"  REFUSED  {item['name']}: {exc}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
