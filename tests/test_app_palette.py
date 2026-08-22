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


def test_normalise_is_what_the_validator_compares_on():
    """The colour rule is a set membership test, so both sides have to be in
    the same form or every drawing is refused for using its own palette."""
    assert palette.normalise("#FFF") == "#ffffff"
    assert palette.normalise("E2703A") == "#e2703a"
    assert palette.normalise("#E2703AFF") == "#e2703a"
