"""Round 24 — contrast is measured, not assumed.

DESIGN_SKILL.md states the WCAG numbers four times — body text ≥ 4.5:1,
large text (24px, or 18.66px bold) ≥ 3:1 — and nothing in the pipeline
measured either one. The founder's recorded build shipped a headline the
VISION model had to catch ("The headline 'Tennis, pocket-sized.' has
insufficient contrast"): the expensive pass doing the free pass's job. The
render gate now computes the ratio inside the same evaluate payload that
measures tap targets, and a failing pair refuses a publish exactly like an
undersized D-pad.

Everything here runs dry: the maths against known pairs through the python
mirror, the thresholds and wording through `layout_findings`, and the parts
only a browser can execute through source probes on the JS itself — the same
split the rest of this gate's suite uses, and for the same reason: both ways
of getting the browser half wrong are silent.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import shutil
import subprocess
import tempfile

import pytest

from app.agent.skills.builtins.app_html import verify


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def contrast_findings(items):
    """The contrast findings alone, out of a layout measurement."""
    return [f for f in verify.layout_findings({"contrast": items})
            if ":1" in f.message]


# ── 1. The ratio math, pinned to known pairs ──────────────────────────

@pytest.mark.parametrize("fg,bg,expected", [
    ("#6B6257", "#F5EFE4", 5.23),   # warm grey on cream — passes 4.5
    ("#8A7E70", "#F5EFE4", 3.46),   # the washed-out version — fails 4.5
    ("#8A93AC", "#0E1424", 5.99),   # slate on near-black — passes
    ("#5D6478", "#18203A", 2.72),   # the recorded headline's class — fails
])
def test_the_ratio_math_matches_known_pairs(fg, bg, expected):
    assert round(verify.contrast_ratio(fg, bg), 2) == expected


def test_the_ratio_is_symmetric_and_bounded():
    # (L1+0.05)/(L2+0.05) with L1 the lighter: order must not matter, the
    # floor is 1:1 (a colour against itself) and the ceiling 21:1 (ink).
    assert verify.contrast_ratio("#8A7E70", "#F5EFE4") == pytest.approx(
        verify.contrast_ratio("#F5EFE4", "#8A7E70")
    )
    assert verify.contrast_ratio("#123456", "#123456") == pytest.approx(1.0)
    assert verify.contrast_ratio("#000000", "#FFFFFF") == pytest.approx(21.0)


def test_the_python_mirror_and_the_js_share_their_constants():
    # The mirror exists so a test can look at maths a browser hides. That is
    # only worth anything while the two are the SAME maths, so every sRGB
    # constant must appear in both.
    py = inspect.getsource(verify.contrast_ratio)
    for const in ("0.2126", "0.7152", "0.0722", "0.03928",
                  "12.92", "0.055", "1.055", "2.4"):
        assert const in py, const
        assert const in verify._LAYOUT_JS, const


# ── 2. The thresholds, testable without a browser ─────────────────────

def test_normal_text_is_held_to_four_point_five():
    failing = contrast_findings(
        [{"label": "Score 12", "ratio": 3.46, "large": False, "lightText": False}]
    )
    assert len(failing) == 1
    assert "3.46:1" in failing[0].message
    assert "4.5:1" in failing[0].message
    passing = contrast_findings(
        [{"label": "Score 12", "ratio": 5.23, "large": False, "lightText": False}]
    )
    assert passing == []


def test_large_text_is_held_to_three():
    # 3.46 fails body text (above) but PASSES a 24px headline — a gate that
    # held headlines to 4.5 would refuse half the working palettes shipped.
    passing = contrast_findings(
        [{"label": "Tennis, pocket-sized.", "ratio": 3.46, "large": True,
          "lightText": True}]
    )
    assert passing == []
    failing = contrast_findings(
        [{"label": "Tennis, pocket-sized.", "ratio": 2.72, "large": True,
          "lightText": True}]
    )
    assert len(failing) == 1
    assert "2.72:1" in failing[0].message
    assert "3:1" in failing[0].message


def test_at_most_two_findings_worst_first():
    out = contrast_findings([
        {"label": "third", "ratio": 3.9, "large": False, "lightText": False},
        {"label": "first", "ratio": 1.4, "large": False, "lightText": False},
        {"label": "second", "ratio": 2.7, "large": False, "lightText": False},
    ])
    assert len(out) == 2
    assert "first" in out[0].message and "1.40:1" in out[0].message
    assert "second" in out[1].message and "2.70:1" in out[1].message
    assert not any("third" in f.message for f in out)


# ── 3. The finding is something a model can act on ────────────────────

def test_the_finding_names_the_text_and_the_direction_to_move():
    light = contrast_findings(
        [{"label": "Tennis, pocket-sized.", "ratio": 2.72, "large": False,
          "lightText": True}]
    )[0]
    assert "Tennis, pocket-sized." in light.message
    assert "Lighten the text or darken the ground" in light.message
    dark = contrast_findings(
        [{"label": "How to play", "ratio": 3.1, "large": False,
          "lightText": False}]
    )[0]
    assert "Darken the text or lighten the ground" in dark.message


def test_the_finding_names_no_internals_and_no_css_machinery():
    # The direction is colour words, never property names — "set
    # background-color" reads as machinery, and half the fixes are not that
    # property anyway (the ground may be a parent, a card, a body).
    out = contrast_findings([
        {"label": "a", "ratio": 2.0, "large": False, "lightText": True},
        {"label": "b", "ratio": 2.5, "large": True, "lightText": False},
    ])
    assert out
    for f in out:
        for banned in ("background-color", "background-image", "rgb(",
                       "getComputedStyle", "luminance", "WCAG", "opts.",
                       "app_html"):
            assert banned not in f.message, (banned, f.message)


def test_contrast_rides_the_same_report_as_an_undersized_dpad():
    # The wiring claim: one measurement dict, one findings list, one report —
    # so a washed-out headline refuses a publish exactly like a 32px D-pad.
    findings = verify.layout_findings({
        "controls": 2,
        "small": [{"label": "D-pad up", "w": 32, "h": 32}],
        "contrast": [{"label": "Tennis, pocket-sized.", "ratio": 2.72,
                      "large": True, "lightText": True}],
    })
    assert any("32×32" in f.message for f in findings)
    assert any("2.72:1" in f.message for f in findings)
    assert all(f.kind == "layout" for f in findings)
    assert not verify.Report(findings=findings).ok


def test_contrast_is_formatted_between_tiny_text_and_overflow():
    # Order probe, same class as the suite's `_press_start` one: the report
    # reads size → contrast → overflow, text problems together.
    src = inspect.getsource(verify.layout_findings)
    assert src.index('m.get("tiny")') < src.index('m.get("contrast")')
    assert src.index('m.get("contrast")') < src.index('m.get("overflowPx")')


def test_the_findings_ride_the_same_extension_as_the_rest_of_layout():
    # ORDER, not presence (the guard class CLAUDE.md records): the layout list
    # — contrast now inside it — must be extended into the report AFTER the
    # blocked-CDN downgrade, or a page whose stylesheet never arrived would be
    # blamed for the geometry of a page that never painted.
    src = inspect.getsource(verify._smoke)
    i_measure = src.index("_measure_layout(page)")
    i_blocked = src.index("if blocked:")
    i_extend = src.index("report.findings.extend(layout)")
    assert i_measure < i_blocked < i_extend


# ── 4. The browser half, probed at the source ─────────────────────────

def test_the_walk_refuses_an_image_before_it_trusts_a_colour():
    # A wrong ratio is worse than none: any background-image on the walk (a
    # gradient computes as one) makes the ground unknowable, so the image
    # check must come before the colour is even parsed — per node, every node.
    js = verify._LAYOUT_JS
    walk = js[js.index("effectiveBg"):js.index("for (const el of")]
    assert walk.index("backgroundImage") < walk.index("backgroundColor")
    # …and a walk that never resolves (text over a canvas or an <img>)
    # answers null, which the caller reads as "do not judge this element".
    assert "return null" in walk
    assert js.count("if (!bg) continue") == 1


def test_translucent_grounds_are_composited_not_read_raw():
    # An rgba( , , ,0.9) card over a dark page is nearly the card's colour,
    # not the page's — reading either raw colour would misjudge both ways.
    js = verify._LAYOUT_JS
    assert "1 - top.a" in js
    assert js.index("const over") < js.index("effectiveBg")


def test_the_js_dedupes_by_colour_pair_and_caps_what_it_sends():
    # One palette mistake paints the same pair on forty elements; forty
    # findings would bury the D-pad. Worst example per pair, then the cap.
    js = verify._LAYOUT_JS
    assert "byPair" in js
    assert "a.ratio - b.ratio" in js
    assert "slice(0, opts.maxContrast)" in js


def test_the_measure_call_hands_the_js_its_thresholds():
    # One source of truth for the numbers: the python constants ride the opts
    # payload, so the JS can never drift to different thresholds silently.
    src = inspect.getsource(verify._measure_layout)
    for name in ("MIN_CONTRAST_NORMAL", "MIN_CONTRAST_LARGE",
                 "LARGE_TEXT_PX", "LARGE_BOLD_PX", "MAX_CONTRAST_FINDINGS"):
        assert name in src, name


@pytest.mark.skipif(not shutil.which("node"), reason="node is the parser here")
def test_the_layout_js_still_parses():
    # The payload is a string python never executes — a stray brace ships
    # green and kills the whole layout gate at evaluate time, silently
    # (its except returns []). Same parser the syntax gate itself uses.
    with tempfile.TemporaryDirectory(prefix="toup-layoutjs-") as tmp:
        path = os.path.join(tmp, "layout.js")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("const f = " + verify._LAYOUT_JS.strip() + ";\n")
        proc = subprocess.run(
            [shutil.which("node"), "--check", path],
            capture_output=True, text=True, timeout=verify.SYNTAX_TIMEOUT_S,
        )
        assert proc.returncode == 0, proc.stderr


# ── 5. Measurement never fails the run ────────────────────────────────

def test_a_measurement_that_cannot_run_is_not_a_verdict():
    class _DeadPage:
        async def evaluate(self, *_a):
            raise RuntimeError("browser crashed mid-read")

    assert run(verify._measure_layout(_DeadPage())) == []


def test_a_malformed_measurement_is_not_a_verdict():
    class _WeirdPage:
        async def evaluate(self, *_a):
            return {"controls": 3, "contrast": [{"label": "x"}]}  # no ratio

    # A missing ratio reads as 0 — below every bar — rather than raising in
    # the formatter; the report stays a report.
    findings = run(verify._measure_layout(_WeirdPage()))
    assert all(isinstance(f, verify.Finding) for f in findings)
