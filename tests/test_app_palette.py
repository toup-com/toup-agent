"""The app's colours come from the app.

Round 20, logo correction. The first icons picked colour from
``_HUES[sha256(slug)[0] % 12]`` — a twelve-hue ramp with no relationship to
anything. A mole game whose every screen is dark green and burnt orange got a
tile in whatever hue its slug happened to hash to, and a library of twenty
apps read as a bag of sweets rather than a shelf of that person's things.

An app already HAS a palette: the model chose one writing the CSS, and the
design skill asks for it as custom properties on ``:root``. These tests are
about reading it back faithfully — and about the case that matters most,
which is having none, because an app with no colours of its own must not
acquire some here.
"""

from __future__ import annotations

import pytest

from app.agent.skills.builtins.app_html import palette


def _styled(css: str) -> str:
    return f"<!doctype html><html><head><style>{css}</style></head><body></body></html>"


# ── Custom properties are the palette ─────────────────────────────────

def test_design_tokens_win_and_keep_their_order():
    """Declaration order is the author's order of importance, so it is kept
    rather than sorted — `--bg` first means the background matters most."""
    html = _styled(":root{--bg:#1E2E1C;--soil:#3E2C1E;--mole:#E2703A;--ink:#F4F1E6;}")
    assert palette.extract(html) == ["#1e2e1c", "#3e2c1e", "#e2703a", "#f4f1e6"]


def test_shorthand_hex_is_expanded():
    assert palette.extract(_styled(":root{--a:#fff;--b:#036;--c:#E2703A}")) == [
        "#ffffff", "#003366", "#e2703a"]


def test_the_palette_is_capped():
    css = ":root{" + ";".join(f"--c{i}:#{i:02x}{i:02x}{i:02x}" for i in range(12)) + "}"
    assert len(palette.extract(_styled(css))) == palette.MAX_PALETTE


def test_a_duplicate_token_is_one_colour():
    html = _styled(":root{--bg:#1E2E1C;--panel:#1e2e1c;--ink:#F4F1E6}")
    assert palette.extract(html) == ["#1e2e1c", "#f4f1e6"]


# ── Literals, for an app that hard-coded everything ───────────────────

def test_literals_are_ranked_by_how_often_they_are_used():
    """A colour used twelve times is a theme; one used once is a detail."""
    html = _styled("body{background:#0B0B0F;color:#F4F4F5}"
                   "a{color:#FF5C39}b{color:#FF5C39}i{color:#FF5C39}")
    assert palette.extract(html)[0] == "#ff5c39"


def test_tokens_beat_literals():
    html = _styled(":root{--bg:#1E2E1C;--ink:#F4F1E6}"
                   "p{color:#123456}p{color:#123456}p{color:#123456}")
    assert palette.extract(html) == ["#1e2e1c", "#f4f1e6"]


# ── Having none is a real answer ──────────────────────────────────────

def test_an_app_with_no_colours_has_no_palette():
    """`[]`, never a default. The caller must treat this as "do not draw" —
    an invented palette is exactly the defect this module removes."""
    assert palette.extract(_styled("body{margin:0;font-family:system-ui}")) == []
    assert palette.extract("") == []
    assert palette.extract("<html><body><p>hi</p></body></html>") == []


def test_one_colour_is_not_a_palette():
    assert palette.extract(_styled("body{background:#0B0B0F}")) == []


def test_black_and_white_alone_still_count():
    """An app really can be black and white; the neutral filter must not
    leave it with nothing when that is all it has."""
    html = _styled("body{background:#000000;color:#ffffff}"
                   "p{color:#ffffff}i{background:#000000}")
    assert palette.extract(html) == ["#000000", "#ffffff"]


# ── Ordering, which the prompt assigns roles by ───────────────────────

def test_ordering_is_by_value_not_declaration():
    """The prompt says "darkest as the ground, lightest for the detail that
    pops", so the roles have to be computed. `--ink` being declared last does
    not make it the lightest."""
    ranked, dark, light = palette.ordered(["#F4F1E6", "#1E2E1C", "#E2703A"])
    assert dark == "#1E2E1C"
    assert light == "#F4F1E6"
    assert ranked == ["#1E2E1C", "#E2703A", "#F4F1E6"]


def test_ordering_an_empty_palette_is_empty():
    assert palette.ordered([]) == ([], "", "")


def test_luminance_puts_the_obvious_pairs_in_the_obvious_order():
    assert palette.luminance("#000000") < palette.luminance("#808080")
    assert palette.luminance("#808080") < palette.luminance("#ffffff")


# ── Round 25: which colour plays which part ───────────────────────────
#
# `ordered` assigns roles by luminance alone — `ranked[0]`, `ranked[-1]`, and
# "a mid colour" for the subject. Whether the app's one accent got the glyph
# depended entirely on where it happened to sort, and on a dark app it did
# not: the middle by luminance is `--muted`, so the accent appeared nowhere
# in the app's own icon and the mark came out grey on near-black.

#: The design skill's own two worked examples (DESIGN_SKILL.md §1d), in the
#: order `extract` returns them.
LIGHT = ["#FFF8F0", "#FFFFFF", "#1A1410", "#7A6A5D", "#F0552B"]   # gym log
DARK = ["#0E1424", "#18203A", "#E8EAF2", "#8A93AC", "#E3A857"]    # sleep aid


def test_the_dark_app_s_accent_finally_gets_the_glyph():
    """The defect, as one assertion. The middle colour by luminance is
    `--muted` (#8A93AC), which is why this app's mark was grey on navy."""
    assert palette.ordered(DARK)[0][len(DARK) // 2] == "#8A93AC"   # the old pick
    parts = palette.roles(DARK)
    assert parts.ground == "#0E1424"        # --bg, still the darkest
    assert parts.glyph == "#E3A857"         # --accent, the chromatic one
    assert parts.detail == "#E8EAF2"        # --ink, which pops on the accent


def test_the_light_app_keeps_the_answer_it_already_had_by_luck():
    """The change must not be a different arbitrary rule. Where luminance
    happened to be right, the new rule agrees with it."""
    assert palette.ordered(LIGHT)[0][len(LIGHT) // 2] == "#F0552B"
    parts = palette.roles(LIGHT)
    assert (parts.ground, parts.glyph) == ("#1A1410", "#F0552B")


def test_the_glyph_is_the_most_chromatic_colour_not_the_middle_one():
    """Directly: two colours sort between the extremes, one grey and one
    saturated, and the saturated one is the subject however they order."""
    for pal in (["#101010", "#7B7B7B", "#D9451F", "#F2F2F2"],
                ["#101010", "#D9451F", "#7B7B7B", "#F2F2F2"]):
        assert palette.roles(pal).glyph == "#D9451F"


def test_a_glyph_that_could_not_be_read_loses_to_one_that_can():
    """Chroma is a preference, not an override. An accent too close to the
    ground to make out at 24px is demoted to the detail — a mark nobody can
    see is worse than a mark that is not in the brand colour."""
    # #241E14 on #1A1410 is the most chromatic colour here and measures 1.3:1.
    pal = ["#1A1410", "#241E14", "#EFEAE2"]
    parts = palette.roles(pal)
    assert palette.chroma("#241E14") > palette.chroma("#EFEAE2")
    assert palette.contrast("#241E14", "#1A1410") < palette.MIN_GLYPH_CONTRAST
    assert parts.glyph == "#EFEAE2"
    assert parts.detail == "#241E14"


def test_the_detail_is_never_the_ground_wearing_another_name():
    """Otherwise "the one detail that must pop" is invisible whenever it
    lands on the ground rather than on the subject."""
    pal = ["#0E1424", "#101627", "#E3A857", "#E8EAF2"]
    parts = palette.roles(pal)
    assert parts.glyph == "#E3A857"
    assert palette.contrast("#101627", parts.ground) < palette.MIN_DETAIL_CONTRAST
    assert parts.detail == "#E8EAF2"


def test_the_ground_is_still_the_darkest_colour():
    """Round 20's decision, kept on purpose: it is what makes a library of
    tiles read as one shelf rather than a bag of sweets. Only the glyph and
    detail roles changed in round 25."""
    for pal in (LIGHT, DARK, ["#F2F2F2", "#D9451F", "#101010"]):
        assert palette.roles(pal).ground == palette.ordered(pal)[1]


def test_roles_survives_a_palette_too_small_to_have_three():
    assert palette.roles([]) == ([], "", "", "")
    assert palette.roles(["#123456"]) == (["#123456"], "#123456", "#123456",
                                          "#123456")
    two = palette.roles(["#101010", "#D9451F"])
    assert (two.ground, two.glyph, two.detail) == ("#101010", "#D9451F",
                                                   "#D9451F")
    # A palette that is one colour written twice is a palette of one.
    same = palette.roles(["#101010", "#101010"])
    assert same.glyph == "#101010"


def test_roles_is_pure_and_repeatable():
    """It decides what a stored icon is painted in, so a second call on a
    second boot has to reach the same answer — and it must not reorder the
    caller's list underneath it."""
    given = list(DARK)
    assert palette.roles(given) == palette.roles(given)
    assert given == DARK


def test_contrast_is_the_wcag_ratio_not_the_ordering_number():
    """`luminance` skips the sRGB transfer curve, which is fine for sorting
    and wrong by a wide margin for a threshold. Burnt orange on dark green is
    2.7 by the naive sum and 4.6 by the real formula — one side of 3.0 each."""
    naive = ((palette.luminance("#E2703A") + 0.05)
             / (palette.luminance("#1E2E1C") + 0.05))
    assert naive < palette.MIN_GLYPH_CONTRAST
    assert palette.contrast("#E2703A", "#1E2E1C") > palette.MIN_GLYPH_CONTRAST
    assert palette.contrast("#000000", "#ffffff") == pytest.approx(21.0, abs=0.01)
    assert palette.contrast("#123456", "#123456") == pytest.approx(1.0)


def test_chroma_separates_the_accent_from_the_greys():
    assert palette.chroma("#808080") == 0.0
    assert palette.chroma("#E3A857") > palette.chroma("#8A93AC")
    assert palette.chroma("#F0552B") > palette.chroma("#FFF8F0")


def test_normalise_is_what_the_validator_compares_on():
    """The colour rule is a set membership test, so both sides have to be in
    the same form or every drawing is refused for using its own palette."""
    assert palette.normalise("#FFF") == "#ffffff"
    assert palette.normalise("E2703A") == "#e2703a"
    assert palette.normalise("#E2703AFF") == "#e2703a"
