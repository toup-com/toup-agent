"""Round 20, item 4 — every app has a mark that says what it is.

Three things have to hold, and the third is the one that would be a security
bug rather than a cosmetic one:

1. **There is always an icon.** A card that renders a broken image is worse
   than a card that renders a plain one, and "it arrives later" is not
   something a list view can express. The designed mark needs a model; the
   monogram needs nothing, so it is the floor.
2. **A placeholder does not become permanent.** A fallback that never
   upgrades looks exactly like success, which is the failure mode this item
   exists to fix — twenty identical tiles.
3. **What the model draws is not trusted.** An SVG is a document that can
   carry script, event handlers and external references, and this one is
   served from an origin the shell trusts. Every one of those is REFUSED, not
   stripped: a mark that had to be edited to be safe is a mark nobody has
   looked at.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from app.agent.skills.builtins.app_html import logo, store
from app.agent.skills.builtins.app_html.logo import IconError

#: A mark drawn to the ROUND 25 spec, and every rule in this file is checked
#: against it. It was rewritten for round 25 and the diff is the whole item:
#: it used to be `q…t…` (relative commands, unmeasurable) drawing a subject
#: from 24 to 94 in a 96 frame — i.e. running off the right edge, which is
#: what round 20 asked for and what made the library a set of blobs. Now:
#: absolute commands only, everything inside 14–82, centred on 48,48.
GOOD = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96" width="96" '
    'height="96"><rect width="96" height="96" fill="#2F6B3A"/>'
    '<path d="M20 62 Q34 40 48 54 Q62 68 76 46" stroke="#F7F4EC" '
    'stroke-width="10" fill="none" stroke-linecap="round"/>'
    '<circle cx="72" cy="42" r="7" fill="#C4703A"/></svg>'
)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture()
def apps_dir(tmp_path, monkeypatch):
    root = tmp_path / "apps"
    monkeypatch.setenv("TOUP_HTML_APPS_DIR", str(root))
    monkeypatch.setenv("TOUP_APP_MODEL_CALLS", "0")
    store.ensure_root()
    return root


# ── 1. There is always an icon ────────────────────────────────────────

def test_a_container_with_no_model_still_gets_a_mark(apps_dir):
    svg, source = _run(logo.ensure_icon("snake", title="Nokia Snake Classic"))
    assert source == "fallback"
    assert svg.startswith("<svg")
    assert logo.sanitize_svg(svg) == svg          # the floor is a valid icon
    assert Path(logo.icon_path("snake")).is_file()


def test_the_holding_mark_wears_the_app_s_own_colours(apps_dir):
    """The correction, at the floor.

    The first fallback was a monogram on a square whose hue came from
    `hash(slug) % 12`. A tile whose colour has no relationship to the app it
    opens is worse than a plain one — and twenty of them made the library
    look like a bag of sweets. The holding mark now uses the app's palette
    and nothing else.
    """
    pal = ["#1E2E1C", "#E2703A", "#F4F1E6"]
    svg = logo.fallback_icon("whack", "Whack a Mole", pal)
    for colour in pal:
        assert colour.lower() in svg.lower(), svg
    # Deterministic: the same app must not change colour between two opens.
    assert svg == logo.fallback_icon("whack", "Whack a Mole", pal)


def test_no_hue_is_invented_for_an_app_that_has_none(apps_dir):
    """An app with no palette does not ACQUIRE one here. Two apps with no
    colours of their own get the same neutral holding mark, which is honest;
    two different invented hues would not be."""
    a = logo.fallback_icon("alpha", "Alpha", [])
    b = logo.fallback_icon("beta", "Beta", [])
    assert a == b
    assert "hsl(" not in a


def test_the_holding_mark_carries_no_lettering(apps_dir):
    """A monogram is a placeholder glyph, which is the thing being removed."""
    svg = logo.fallback_icon("nokia-snake", "Nokia Snake", ["#1B2410", "#9BB53F"])
    assert "<text" not in svg
    assert not hasattr(logo, "initials")
    logo.sanitize_svg(svg)          # and it passes the real validator


def test_the_icon_lives_where_the_library_cannot_see_it(apps_dir):
    _run(logo.ensure_icon("snake", title="Snake"))
    path = Path(logo.icon_path("snake"))
    assert path.parent.name == ".icons"           # a dot-directory
    assert path.suffix == ".svg"                  # not the listed suffix
    assert path.parent.parent == Path(store.apps_root())   # below depth 0

    from app.services import library_service as lib
    store.write_app("snake", "Snake", "<!doctype html><html><head></head>"
                    "<body><p>" + "x" * 500 + "</p></body></html>")
    keys = [c.key for c in lib._iter_dir_files(store.apps_root(), lib.ROOT_APP, "",
                                               recursive_depth=0, budget=[999])]
    assert not any(".icons" in k for k in keys), keys


# ── 2. A placeholder does not become permanent ────────────────────────

def test_generation_1_icons_are_all_stale(apps_dir):
    """How all 22 existing apps get redrawn: bump `ICON_GENERATION`.

    Every icon on every volume was drawn by the art direction the correction
    rejects, so every one of them has to go — and the mechanism is the same
    self-heal that backfills the briefs, not a migration and not a sweep.
    """
    logo._store_icon("snake", GOOD, source="model", title="Snake",
                     purpose="a snake game")
    assert not logo.is_stale("snake", title="Snake", purpose="a snake game")

    import json as _json
    meta = logo.read_sidecar("snake")
    meta["gen"] = "1"                                   # as generation 1 left it
    with open(logo.sidecar_path("snake"), "w", encoding="utf-8") as fh:
        _json.dump(meta, fh)
    assert logo.is_stale("snake", title="Snake", purpose="a snake game")


def test_a_fallback_is_always_stale(apps_dir):
    """So the first run that CAN reach a model replaces it. Without this, a
    container that was briefly offline keeps a monogram forever — and that
    reads as success."""
    _run(logo.ensure_icon("snake", title="Snake"))
    assert logo.read_sidecar("snake")["source"] == "fallback"
    assert logo.is_stale("snake", title="Snake", purpose="")


def test_a_designed_mark_is_kept(apps_dir):
    logo._store_icon("snake", GOOD, source="model", title="Snake",
                     purpose="a snake game")
    assert not logo.is_stale("snake", title="Snake", purpose="a snake game")
    svg, source = _run(logo.ensure_icon("snake", title="Snake",
                                        purpose="a snake game"))
    assert source == "kept"
    assert svg == GOOD


def test_a_failed_redraw_never_downgrades_a_good_mark(apps_dir):
    """One unreachable model must not cost an app the icon it already has."""
    logo._store_icon("snake", GOOD, source="model", title="Snake",
                     purpose="a snake game")
    # A rename makes it stale; the model cannot be reached to redraw it.
    svg, source = _run(logo.ensure_icon("snake", title="Snake Deluxe",
                                        purpose="a snake game"))
    assert source == "kept"
    assert svg == GOOD


def test_the_mark_is_redrawn_when_the_app_changes_what_it_is(apps_dir):
    logo._store_icon("app", GOOD, source="model", title="Timer",
                     purpose="a pomodoro timer")
    assert not logo.is_stale("app", title="Timer", purpose="a pomodoro timer")
    assert logo.is_stale("app", title="Timer", purpose="a budget tracker")
    assert logo.is_stale("app", title="Budget", purpose="a pomodoro timer")


def test_the_mark_is_not_redrawn_for_an_ordinary_edit(apps_dir):
    """Regenerating on every edit spends a model call on a padding change and
    makes the tile flicker between revisions."""
    logo._store_icon("app", GOOD, source="model", title="Timer",
                     purpose="a pomodoro timer for focused work")
    for _ in range(5):
        assert not logo.is_stale("app", title="Timer",
                                 purpose="a pomodoro timer for focused work")


# ── 3. What the model draws is not trusted ────────────────────────────

def test_a_good_drawing_is_accepted():
    assert logo.sanitize_svg(GOOD) == GOOD
    assert logo.sanitize_svg(f"```svg\n{GOOD}\n```") == GOOD
    assert logo.sanitize_svg(f"Here is the icon:\n{GOOD}\nHope that helps!") == GOOD


@pytest.mark.parametrize("svg,why", [
    (GOOD.replace("<circle", "<script>alert(1)</script><circle"), "script"),
    (GOOD.replace("<rect ", '<rect onload="alert(1)" '), "event attribute"),
    (GOOD.replace("<circle", '<image href="https://evil.test/x.png"/><circle'),
     "external reference"),
    (GOOD.replace("<circle", '<image href="data:image/png;base64,AAA"/><circle'),
     "embedded file"),
    (GOOD.replace("<circle",
                  '<foreignObject><div>hi</div></foreignObject><circle'),
     "foreignObject"),
    (GOOD.replace(' viewBox="0 0 96 96"', ""), "no viewBox"),
])
def test_a_drawing_that_is_not_only_drawing_is_refused(svg, why):
    with pytest.raises(IconError):
        logo.sanitize_svg(svg)


def test_a_stub_and_a_photograph_are_both_refused():
    with pytest.raises(IconError):
        logo.sanitize_svg('<svg viewBox="0 0 1 1"></svg>')
    with pytest.raises(IconError):
        logo.sanitize_svg(
            GOOD.replace("</svg>", "<path d='" + "M0 0 " * 20000 + "'/></svg>")
        )
    with pytest.raises(IconError):
        logo.sanitize_svg("not an svg at all")
    with pytest.raises(IconError):
        logo.sanitize_svg("")


def test_refusal_names_what_is_wrong():
    """The message goes back to a model that can draw it again."""
    with pytest.raises(IconError) as exc:
        logo.sanitize_svg(GOOD.replace("<circle", "<script>x</script><circle"))
    assert "script" in str(exc.value)


# ── The route ─────────────────────────────────────────────────────────

def test_the_icon_route_never_404s_for_an_app_that_exists(apps_dir):
    """A list view cannot express "the icon arrives later"."""
    from app.api import artifacts as routes

    store.write_app("snake", "Nokia Snake",
                    "<!doctype html><html><head></head><body><p>"
                    + "x" * 500 + "</p></body></html>")
    assert logo.read_icon("snake") is None       # nothing drawn yet

    class _Req:
        headers: dict = {}

    resp = _run(routes.get_artifact_icon("snake", _Req()))
    assert resp.status_code == 200
    assert resp.media_type == "image/svg+xml"
    assert resp.body.decode().startswith("<svg")
    assert resp.headers["x-toup-icon-source"] == "fallback"
    # nosniff, because this is a model-authored document served from an
    # origin the shell trusts.
    assert resp.headers["x-content-type-options"] == "nosniff"


def test_the_icon_route_revalidates_instead_of_re_sending(apps_dir):
    from app.api import artifacts as routes

    store.write_app("snake", "Nokia Snake",
                    "<!doctype html><html><head></head><body><p>"
                    + "x" * 500 + "</p></body></html>")

    class _Req:
        def __init__(self, etag=None):
            self.headers = {"if-none-match": etag} if etag else {}

    first = _run(routes.get_artifact_icon("snake", _Req()))
    etag = first.headers["etag"]
    again = _run(routes.get_artifact_icon("snake", _Req(etag)))
    assert again.status_code == 304
    assert not again.body


def test_an_unknown_app_has_no_icon(apps_dir):
    from fastapi import HTTPException
    from app.api import artifacts as routes

    class _Req:
        headers: dict = {}

    with pytest.raises(HTTPException) as exc:
        _run(routes.get_artifact_icon("ghost", _Req()))
    assert exc.value.status_code == 404


def test_the_list_says_whether_an_icon_is_worth_fetching(apps_dir):
    from app.api import artifacts as routes

    store.write_app("snake", "Nokia Snake",
                    "<!doctype html><html><head></head><body><p>"
                    + "x" * 500 + "</p></body></html>")
    listing = _run(routes.list_artifacts())
    row = listing["apps"][0]
    assert row["has_icon"] is False              # nothing drawn yet
    _run(logo.ensure_icon("snake", title="Nokia Snake"))
    assert _run(routes.list_artifacts())["apps"][0]["has_icon"] is True
    # ...and never the drawing itself: this payload rides every list.
    assert "<svg" not in str(listing)


def test_deleting_an_app_takes_its_icon(apps_dir):
    store.write_app("snake", "Nokia Snake",
                    "<!doctype html><html><head></head><body><p>"
                    + "x" * 500 + "</p></body></html>")
    _run(logo.ensure_icon("snake", title="Nokia Snake"))
    assert logo.read_icon("snake") is not None
    store.delete_app("snake")
    assert logo.read_icon("snake") is None
    assert not os.path.exists(logo.sidecar_path("snake"))


def test_both_icon_models_are_pinned():
    """`model=None` resolves to the tenant's CHAT model, and this runs once
    per app on a background sweep over the whole library."""
    for model in (logo.SUBJECT_MODEL, logo.DRAW_MODEL):
        assert model and model != "None"


# ── 4. The colour rule ────────────────────────────────────────────────

def test_a_colour_from_outside_the_palette_is_refused():
    """The strictest rule in the file, and the one the correction turns on.

    The first icons chose colour from a hash of the slug, so a mole game with
    a dark-green and burnt-orange screen got a tile in whatever hue its slug
    landed on. The palette is now read from the app and enforced here — not
    suggested to the model and hoped for.
    """
    pal = ["#2F6B3A", "#F7F4EC", "#C4703A"]
    assert logo.sanitize_svg(GOOD, pal) == GOOD          # all three are in it
    with pytest.raises(IconError) as exc:
        logo.sanitize_svg(GOOD.replace("#C4703A", "#FF00FF"), pal)
    assert "#ff00ff" in str(exc.value).lower()
    assert "palette" in str(exc.value)


def test_the_palette_rule_is_only_applied_when_there_is_a_palette():
    """An app with no colours of its own must not have some invented for it,
    and must not be refused for using any."""
    assert logo.sanitize_svg(GOOD) == GOOD
    assert logo.sanitize_svg(GOOD, []) == GOOD


def test_gradients_and_none_are_not_stray_colours():
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96">'
        '<defs><linearGradient id="g"><stop offset="0" stop-color="#2F6B3A"/>'
        '<stop offset="1" stop-color="#C4703A"/></linearGradient></defs>'
        '<rect width="96" height="96" fill="url(#g)"/>'
        '<circle cx="48" cy="48" r="30" fill="#F7F4EC"/>'
        '<path d="M20 74 L76 74 L48 24 Z" fill="none" stroke="#C4703A" '
        'stroke-width="9"/></svg>')
    assert logo.sanitize_svg(svg, ["#2F6B3A", "#F7F4EC", "#C4703A"])


# ── 5. It has to look like a mark ─────────────────────────────────────

def test_a_mark_floating_on_transparency_is_refused():
    """Every first-generation icon was a small pictogram on a badge. The
    full-bleed ground is what makes the set one family instead of stickers."""
    no_ground = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96">'
        '<circle cx="48" cy="48" r="24" fill="#2F6B3A"/>'
        '<circle cx="60" cy="40" r="12" fill="#F7F4EC"/>'
        '<circle cx="30" cy="60" r="10" fill="#C4703A"/></svg>')
    with pytest.raises(IconError) as exc:
        logo.sanitize_svg(no_ground)
    assert "full-bleed" in str(exc.value)


def test_a_ground_with_nothing_on_it_is_refused():
    bare = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96">'
            '<rect width="96" height="96" fill="#2F6B3A"/>'
            '<circle cx="48" cy="48" r="20" fill="#F7F4EC"/>'
            + "<!-- " + "x" * 200 + " -->" + '</svg>')
    with pytest.raises(IconError) as exc:
        logo.sanitize_svg(bare)
    assert "nothing on it" in str(exc.value)


def test_an_illustration_is_refused():
    """Thirteen shapes is a lattice or a scene; either is mud at 24px."""
    busy = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96">'
            '<rect width="96" height="96" fill="#2F6B3A"/>'
            + '<circle cx="10" cy="10" r="4" fill="#F7F4EC"/>' * 20 + '</svg>')
    with pytest.raises(IconError) as exc:
        logo.sanitize_svg(busy)
    assert "illustration" in str(exc.value)


def test_a_hairline_is_refused():
    thin = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96">'
            '<rect width="96" height="96" fill="#2F6B3A"/>'
            '<circle cx="48" cy="48" r="30" fill="#F7F4EC"/>'
            '<path d="M10 10 L90 90" stroke="#C4703A" stroke-width="1.5"/>'
            + "<!-- " + "x" * 120 + " -->" + '</svg>')
    with pytest.raises(IconError) as exc:
        logo.sanitize_svg(thin)
    assert "24px" in str(exc.value)


def test_lettering_is_refused():
    """An icon that has to be READ is not an icon — and a monogram is exactly
    the placeholder glyph this correction removes."""
    lettered = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96">'
                '<rect width="96" height="96" fill="#2F6B3A"/>'
                '<circle cx="48" cy="48" r="30" fill="#F7F4EC"/>'
                '<circle cx="60" cy="60" r="10" fill="#C4703A"/>'
                '<text x="48" y="52" font-size="40">NS</text></svg>')
    with pytest.raises(IconError) as exc:
        logo.sanitize_svg(lettered)
    assert "text" in str(exc.value).lower()


# ── 6. No two apps share a symbol ─────────────────────────────────────

def test_the_subjects_already_drawn_are_offered_to_the_next_app(apps_dir):
    store.write_app("a", "A", "<!doctype html><html><head></head><body><p>"
                    + "x" * 500 + "</p></body></html>")
    store.write_app("b", "B", "<!doctype html><html><head></head><body><p>"
                    + "x" * 500 + "</p></body></html>")
    logo._store_icon("a", GOOD, source="model", title="A", subject="mole and mallet")
    logo._store_icon("b", GOOD, source="model", title="B", subject="coins cascading")

    assert set(logo.subjects_in_use()) == {"mole and mallet", "coins cascading"}
    # ...and an app never competes with itself when it is being redrawn.
    assert logo.subjects_in_use(exclude="a") == ["coins cascading"]


# ── 6b. Round 25: the glyph is centred, and that is MEASURED ──────────
#
# The art direction reversed here. Round 20 mandated "HUGE SUBJECT — it fills
# most of the frame and runs off at least one edge, clipped by it. A small
# object centred with space around it is the failure to avoid." That is what
# produced the flat blobs: two to four bold shapes stretched until the frame
# crops them IS a blob. Round 25 asks for the opposite, and — because the
# repo's own lesson is that a validator with teeth beats a longer prompt —
# every clause below is a measurement, and every one of these tests watched
# it reject something.

def _mark(body: str, view: str = "0 0 96 96") -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view}" '
            f'width="96" height="96"><rect width="96" height="96" '
            f'fill="#2F6B3A"/>{body}</svg>')


#: The same subject, drawn once inside the safe area and once bleeding off
#: the frame the way round 20 asked for. Everything else about them is equal.
_CENTRED = _mark('<circle cx="48" cy="48" r="27" fill="#F7F4EC"/>'
                 '<circle cx="59" cy="38" r="10" fill="#C4703A"/>')
_BLEEDING = _mark('<circle cx="48" cy="48" r="52" fill="#F7F4EC"/>'
                  '<circle cx="70" cy="26" r="18" fill="#C4703A"/>')


def test_the_round_20_bleed_is_now_the_refusal():
    """The reversal, as one assertion. The bleeding mark is what generation 2
    was told to draw; it is refused now, and the centred one — identical but
    for its scale and placement — is accepted."""
    assert logo.sanitize_svg(_CENTRED) == _CENTRED
    with pytest.raises(IconError) as exc:
        logo.sanitize_svg(_BLEEDING)
    assert "safe area" in str(exc.value)


def test_every_generation_2_icon_is_stale(apps_dir):
    """The redraw of the whole library, and the reason it is needed: every
    icon on every volume was drawn to the bleed direction."""
    assert logo.ICON_GENERATION == 3
    logo._store_icon("snake", GOOD, source="model", title="Snake",
                     purpose="a snake game")
    import json as _json
    for stale_gen in ("1", "2"):
        meta = logo.read_sidecar("snake")
        meta["gen"] = stale_gen
        with open(logo.sidecar_path("snake"), "w", encoding="utf-8") as fh:
            _json.dump(meta, fh)
        assert logo.is_stale("snake", title="Snake", purpose="a snake game")


@pytest.mark.parametrize("edge,body", [
    ("left", '<rect x="2" y="30" width="60" height="36" fill="#F7F4EC"/>'
             '<circle cx="48" cy="48" r="14" fill="#C4703A"/>'),
    ("right", '<rect x="34" y="30" width="60" height="36" fill="#F7F4EC"/>'
              '<circle cx="48" cy="48" r="14" fill="#C4703A"/>'),
    ("top", '<rect x="30" y="1" width="36" height="60" fill="#F7F4EC"/>'
            '<circle cx="48" cy="48" r="14" fill="#C4703A"/>'),
    ("bottom", '<rect x="30" y="34" width="36" height="61" fill="#F7F4EC"/>'
               '<circle cx="48" cy="48" r="14" fill="#C4703A"/>'),
])
def test_a_subject_clipped_by_any_of_the_four_edges_is_refused(edge, body):
    with pytest.raises(IconError) as exc:
        logo.sanitize_svg(_mark(body))
    assert "safe area" in str(exc.value), edge


def test_a_stroke_paints_outside_its_line_and_that_counts():
    """A centreline at 15 with a 12-unit stroke reaches 9. Measuring the
    geometry and ignoring the stroke would call that centred."""
    ok = _mark('<path d="M22 48 L74 48" stroke="#F7F4EC" stroke-width="12" '
               'fill="none"/>'
               '<circle cx="48" cy="48" r="21" fill="#C4703A"/>')
    assert logo.sanitize_svg(ok) == ok
    fat = _mark('<path d="M15 48 L81 48" stroke="#F7F4EC" stroke-width="24" '
                'fill="none"/>'
                '<circle cx="48" cy="48" r="21" fill="#C4703A"/>')
    with pytest.raises(IconError) as exc:
        logo.sanitize_svg(fat)
    assert "safe area" in str(exc.value)


def test_a_small_mark_adrift_in_the_ground_is_refused():
    """The other half of the reversal. "Not bleeding" is not the goal —
    centred AND filling the safe box is."""
    tiny = _mark('<circle cx="48" cy="48" r="12" fill="#F7F4EC"/>'
                 '<circle cx="52" cy="44" r="5" fill="#C4703A"/>')
    with pytest.raises(IconError) as exc:
        logo.sanitize_svg(tiny)
    assert "96 frame" in str(exc.value)


def test_the_short_axis_floor_is_where_a_bar_becomes_an_object():
    """The number that moved after looking at the contact sheet, pinned here
    so it does not drift back. 18 units was arithmetic — 4.5px of a 24px tile
    — and rendering it showed a dash with no inside. 24 is a shape."""
    def _bar(height: int) -> str:
        y = 48 - height / 2
        return _mark(f'<rect x="14" y="{y}" width="68" height="{height}" '
                     f'fill="#F7F4EC"/>'
                     f'<circle cx="48" cy="48" r="{height / 4:.1f}" '
                     f'fill="#C4703A"/>')

    assert logo.ICON_MIN_GLYPH_SHORT == 24.0
    with pytest.raises(IconError):
        logo.sanitize_svg(_bar(18))
    assert logo.sanitize_svg(_bar(24)) == _bar(24)


def test_a_knockout_in_the_ground_colour_is_not_the_mark_running_off():
    """A crescent, a bite, a gap: all drawn by overpainting in the ground's
    own colour, and the cutting shape routinely reaches past what the mark
    shows. It paints nothing, so it is not where the mark runs — refusing it
    would push the model off the one technique that makes a two-shape subject
    look drawn rather than stamped."""
    crescent = _mark('<circle cx="46" cy="46" r="30" fill="#F7F4EC"/>'
                     '<circle cx="64" cy="34" r="26" fill="#2F6B3A"/>'
                     '<circle cx="30" cy="66" r="6" fill="#C4703A"/>')
    assert logo.sanitize_svg(crescent, ["#2F6B3A", "#F7F4EC", "#C4703A"])
    # The same circle in any OTHER colour paints, so it counts — and it
    # reaches 90, off the right edge.
    painted = crescent.replace('r="26" fill="#2F6B3A"', 'r="26" fill="#C4703A"')
    with pytest.raises(IconError) as exc:
        logo.sanitize_svg(painted, ["#2F6B3A", "#F7F4EC", "#C4703A"])
    assert "safe area" in str(exc.value)


def test_a_subject_shoved_to_one_side_is_refused():
    off = _mark('<circle cx="32" cy="48" r="18" fill="#F7F4EC"/>'
                '<rect x="14" y="28" width="40" height="40" fill="#C4703A"/>')
    with pytest.raises(IconError) as exc:
        logo.sanitize_svg(off)
    assert "off to one side" in str(exc.value)
    # It is inside the safe area and it is big enough — only the placement is
    # wrong, so only the placement rule may speak.
    assert "safe area" not in str(exc.value)


def test_a_relative_path_is_refused_because_it_cannot_be_measured():
    """What makes the geometry measurable at all — and the rule is in the
    draw prompt too, so the model can comply rather than guess."""
    relative = _mark('<path d="M20 62 q14 -22 28 -8 t28 -8" stroke="#F7F4EC" '
                     'stroke-width="10" fill="none"/>'
                     '<circle cx="48" cy="48" r="14" fill="#C4703A"/>')
    with pytest.raises(IconError) as exc:
        logo.sanitize_svg(relative)
    assert "relative" in str(exc.value)
    assert "M, L, C" in str(exc.value)
    # The absolute spelling of a comparable curve is accepted.
    assert logo.sanitize_svg(GOOD) == GOOD


def test_an_exponent_is_not_a_relative_command():
    """`e` is in [a-z] and is not a path command. Refusing it would reject a
    drawing for using scientific notation."""
    exp = _mark('<path d="M20 6.2e1 L76 30 L48 76 Z" fill="#F7F4EC"/>'
                '<circle cx="48" cy="48" r="12" fill="#C4703A"/>')
    assert logo.sanitize_svg(exp) == exp


def test_a_transform_is_refused():
    """Not a safety rule — the coordinates of a transformed shape say nothing
    about where it lands, so the safe area would be measuring fiction."""
    moved = _mark('<circle cx="48" cy="48" r="27" fill="#F7F4EC" '
                  'transform="translate(40 0)"/>'
                  '<circle cx="59" cy="38" r="10" fill="#C4703A"/>')
    with pytest.raises(IconError) as exc:
        logo.sanitize_svg(moved)
    assert "transform" in str(exc.value)


def test_an_arc_is_bounded_by_its_bulge_not_its_endpoints():
    """Two endpoints on the same line do not bound an arc between them — a
    semicircle swings a full radius away from both. Taking the endpoints
    would wave a mark through that runs to the frame edge."""
    bulge = _mark('<path d="M6 48 A42 42 0 1 0 90 48" stroke="#F7F4EC" '
                  'stroke-width="10" fill="none"/>'
                  '<circle cx="48" cy="40" r="12" fill="#C4703A"/>')
    with pytest.raises(IconError) as exc:
        logo.sanitize_svg(bulge)
    assert "safe area" in str(exc.value)
    box = logo.measure_glyph(bulge)
    assert box is not None and box[3] > 85       # the bulge, actually seen


def test_the_path_parser_is_right_about_the_commands_that_carry_state():
    """Every geometry refusal is fiction if this is wrong, so it is pinned
    directly rather than only through the rules built on it. H and V carry one
    coordinate and inherit the other; S and T take an IMPLIED control point
    reflected from the previous curve, and ignoring that reflection turns a
    curve into a straight line — 12 units of extent that were really there."""
    def _bbox(d):
        pts = logo._path_points(d)
        return (min(p[0] for p in pts), min(p[1] for p in pts),
                max(p[0] for p in pts), max(p[1] for p in pts))

    assert _bbox("M20 20 H76 V76 H20 Z") == (20, 20, 76, 76)
    # The T's control point is (34,24) reflected about (48,48) = (62,72), so
    # the second curve swells to y=60. Read as "no reflection" it would stop
    # at 48 and the mark would measure smaller than it draws.
    box = _bbox("M20 48 Q34 24 48 48 T76 48")
    assert box[0] == 20 and box[2] == 76
    assert box[1] == pytest.approx(36, abs=0.5)
    assert box[3] == pytest.approx(60, abs=0.5)
    # …and the same shape written with an explicit Q agrees with it.
    assert _bbox("M20 48 Q34 24 48 48 Q62 72 76 48") == pytest.approx(box, abs=0.01)


def test_the_frame_is_the_viewbox_whatever_its_units():
    """A mark drawn in a 0 0 24 24 viewBox is judged on its composition, not
    refused for its units — otherwise the refusal names the wrong fault and
    the redraw fixes the wrong thing."""
    small = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
             'width="96" height="96"><rect width="24" height="24" '
             'fill="#2F6B3A"/><circle cx="12" cy="12" r="7" fill="#F7F4EC"/>'
             '<circle cx="15" cy="9" r="2.5" fill="#C4703A"/></svg>')
    assert logo.sanitize_svg(small) == small
    bleeding = small.replace('r="7"', 'r="13"')
    with pytest.raises(IconError):
        logo.sanitize_svg(bleeding)


def test_the_ground_may_declare_its_size_in_any_order():
    """A correctness bug, not a security one: the old rule was a regex that
    required width BEFORE height, so a perfectly good ground rect was refused
    and burned one of three redraw attempts on nothing."""
    reversed_attrs = _CENTRED.replace('<rect width="96" height="96"',
                                      '<rect height="96" width="96"')
    assert logo.sanitize_svg(reversed_attrs) == reversed_attrs
    # …and the percentage spelling, which already worked, still does.
    percent = _CENTRED.replace('width="96" height="96" fill="#2F6B3A"',
                               'width="100%" height="100%" fill="#2F6B3A"')
    assert logo.sanitize_svg(percent) == percent


def test_the_composition_rules_do_not_fail_a_drawing_they_cannot_read():
    """Fail open, per the module's contract: a look rule that cannot measure
    has no opinion. The rules that are about SAFETY are not like this."""
    unreadable = _mark('<path d="M Z" fill="#F7F4EC"/>'
                       '<path d="" fill="#C4703A"/>')
    assert logo.measure_glyph(unreadable) is None
    assert logo.sanitize_svg(unreadable) == unreadable
    # And specifically: a Z with no moveto before it does not become a point
    # on the origin, which would refuse the drawing for bleeding off a corner
    # it never touches.
    assert logo._path_points("M Z") == []


# ── 6c. The SVG gate, where it was walked through ─────────────────────

@pytest.mark.parametrize("body,why", [
    ('<animateTransform attributeName="transform" type="rotate" />',
     "animateTransform"),
    ('<animateMotion path="M0 0 L9 9"/>', "animateMotion"),
    ('<style>@import url("//evil.example/x.css");</style>', "style @import"),
    ('<textPath href="#p">hello</textPath>', "textPath"),
    ('<a xlink:href="javascript:alert(1)"><circle cx="48" cy="48" r="4"/></a>',
     "javascript: URI"),
    ('<use href="https://evil.example/x.svg#a"/>', "off-origin use"),
    ('<rect x="30" y="30" width="20" height="20" '
     'style="fill:url(//evil.example/x.svg#f)"/>', "CSS url()"),
])
def test_the_gate_no_longer_has_these_holes(body, why):
    """All four were confirmed by execution before they were fixed:

    * ``animate\\b`` does not match ``<animateTransform`` — the ``\\b``
      between "e" and "T" is not a word boundary, so the whole SMIL family
      except bare ``<animate>`` walked through. Same bug for ``<textPath>``.
    * ``<style>`` was not in the forbidden list at all, and the reference
      rule only looked at href/src, never at CSS ``url()`` — so
      ``@import url("//…")`` was a clean pass.
    * the reference rule wanted ``//`` or ``data:`` specifically, and
      ``javascript:`` is neither.
    """
    with pytest.raises(IconError):
        logo.sanitize_svg(_CENTRED.replace("</svg>", body + "</svg>"))


def test_a_stylesheet_is_refused_even_with_nothing_in_it_to_load():
    """Found by mutation: deleting `style` from the forbidden tags failed no
    test, because the @import case above was being caught by the url() rule
    instead. Both rules are wanted — an icon carries no stylesheet at all,
    and that is the half that holds when a reference is written in a form
    the url() rule does not recognise."""
    styled = _CENTRED.replace(
        "</svg>", "<style>circle{fill:#C4703A}</style></svg>")
    with pytest.raises(IconError):
        logo.sanitize_svg(styled)


def test_a_gradient_is_still_named_by_url_and_that_is_fine():
    """The url() rule has to let through the one url() a drawing needs, or
    every gradient is a refusal."""
    grad = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96">'
            '<defs><linearGradient id="g"><stop offset="0" stop-color="#2F6B3A"/>'
            '<stop offset="1" stop-color="#C4703A"/></linearGradient></defs>'
            '<rect width="96" height="96" fill="url(#g)"/>'
            '<circle cx="48" cy="48" r="28" fill="#F7F4EC"/>'
            '<circle cx="58" cy="38" r="10" fill="#C4703A"/></svg>')
    assert logo.sanitize_svg(grad, ["#2F6B3A", "#F7F4EC", "#C4703A"]) == grad


def test_the_holding_mark_obeys_the_spec_it_is_the_floor_of():
    """A spec its own fallback violates has never been checked end to end.
    Round 20's bands ran x=0 to x=96 — straight through the safe area the
    designed marks are now held to."""
    for pal in ([], ["#1E2E1C", "#E2703A", "#F4F1E6"],
                ["#0E1424", "#18203A", "#E8EAF2", "#8A93AC", "#E3A857"]):
        svg = logo.fallback_icon("x", "X", pal)
        assert logo.sanitize_svg(svg, pal or None) == svg
        box = logo.measure_glyph(svg)
        assert box is not None
        assert box[0] >= logo.ICON_SAFE_MIN and box[2] <= logo.ICON_SAFE_MAX


def test_the_holding_mark_uses_the_app_s_accent():
    """The palette defect one layer away: on a dark app the middle colour by
    luminance is `--muted`, so the accent appeared nowhere and the mark was
    grey on near-black."""
    dark = ["#0E1424", "#18203A", "#E8EAF2", "#8A93AC", "#E3A857"]
    assert "#E3A857" in logo.fallback_icon("sleep", "Sleep", dark)


def test_a_refusal_is_written_to_be_handed_back(apps_dir, monkeypatch):
    """Every new rule's message goes into draw_mark's retry prompt verbatim,
    so it has to name the fault AND the fix."""
    seen = []

    async def _ask(system, user, **_k):
        seen.append(user)
        return _BLEEDING if len(seen) == 1 else GOOD

    monkeypatch.setattr(logo, "_ask", _ask)
    svg = _run(logo.draw_mark(user_id="u", title="Snake", scene="A snake.",
                              palette=["#2F6B3A", "#F7F4EC", "#C4703A"]))
    assert svg == GOOD
    assert "REJECTED" in seen[1]
    assert "safe area" in seen[1]
    assert "14" in seen[1] and "82" in seen[1]


def test_the_draw_prompt_asks_for_what_the_validator_measures():
    """A rule the prompt never states is a rule the model can only discover
    by being refused three times and falling back to bands."""
    prompt = logo._DRAW_SYSTEM
    assert "ABSOLUTE path commands only" in prompt
    assert "14,14 to 82,82" in prompt
    assert "transform=" in prompt
    # …and the round-20 direction is gone from it, not merely contradicted.
    assert "runs off at least one edge" not in prompt
    assert "HUGE SUBJECT" not in prompt


def test_a_slug_cannot_escape_the_icon_directory(apps_dir):
    from app.agent.skills.builtins.app_html.store import AppStoreError

    for bad in ("../outside", "a/b", ".."):
        with pytest.raises(AppStoreError):
            logo.icon_path(bad)


# ── 7. Naming the subject is its own step, and it is checked ──────────

def _subject(monkeypatch, answer, used=()):
    """Run `choose_subject` against a canned model answer."""
    async def _fake(*_a, **_k):
        return answer
    monkeypatch.setattr(logo, "_ask", _fake)
    return _run(logo.choose_subject(user_id="u", title="Pomodoro Timer",
                                    purpose="A 25-minute focus timer.",
                                    used=list(used)))


def test_a_named_subject_is_taken(monkeypatch):
    key, scene = _subject(monkeypatch,
        "KEY: tomato with time wedge\nSCENE: A ripe tomato with one wedge cut away.")
    assert key == "tomato with time wedge"
    assert scene.startswith("A ripe tomato")


def test_a_stock_glyph_is_refused_even_when_the_model_offers_it(monkeypatch):
    """The ban is in the prompt too. This is the half that does not depend on
    the prompt being obeyed — and it was not: asked to choose and draw in one
    breath, the model returned a clock for the timer and a document for the
    budget every single time.
    """
    for stock in ("clock face", "a document", "spreadsheet grid", "map pin",
                  "magnifying glass", "letter p monogram"):
        key, scene = _subject(monkeypatch, f"KEY: {stock}\nSCENE: A {stock}.")
        assert (key, scene) == ("", ""), stock


def test_a_subject_another_app_already_uses_is_refused(monkeypatch):
    key, scene = _subject(monkeypatch,
        "KEY: coins cascading\nSCENE: Coins falling.", used=["coins cascading"])
    assert (key, scene) == ("", "")


def test_an_unparseable_answer_is_not_a_subject(monkeypatch):
    """"" means "do not draw", never "draw something default"."""
    for junk in ("Sure! How about a tomato?", "", "KEY: only a key"):
        assert _subject(monkeypatch, junk) == ("", "")


def test_no_subject_means_no_drawing(apps_dir, monkeypatch):
    """The whole point of naming first: a bad subject costs nothing, because
    nothing is drawn until the subject survives."""
    store.write_app("timer", "Pomodoro Timer",
                    '<!doctype html><html><head><style>:root{--bg:#2B2724;'
                    '--ink:#C1443A;--paper:#F3EDE4}</style></head><body><p>'
                    + "x" * 500 + "</p></body></html>")

    drew = []

    async def _never(**_k):
        drew.append(1)
        return GOOD

    async def _no_subject(**_k):
        return "", ""

    monkeypatch.setenv("TOUP_APP_MODEL_CALLS", "1")
    monkeypatch.setattr(logo, "choose_subject", _no_subject)
    monkeypatch.setattr(logo, "draw_mark", _never)
    from app.agent.skills.builtins.app_html import vision
    monkeypatch.setattr(vision, "can_call_model", lambda: True)

    _svg, source = _run(logo.ensure_icon("timer", title="Pomodoro Timer",
                                         user_id="u"))
    assert source == "fallback"
    assert not drew


def test_an_app_with_no_palette_is_never_drawn_in_invented_colours(apps_dir, monkeypatch):
    """A mark in colours from nowhere is the defect being removed. Better a
    holding mark that says it is one."""
    store.write_app("plain", "Plain", "<!doctype html><html><head><style>"
                    "body{margin:0}</style></head><body><p>" + "x" * 500
                    + "</p></body></html>")
    drew = []

    async def _never(**_k):
        drew.append(1)
        return GOOD

    async def _subject_ok(**_k):
        return "a thing", "A thing."

    monkeypatch.setenv("TOUP_APP_MODEL_CALLS", "1")
    monkeypatch.setattr(logo, "choose_subject", _subject_ok)
    monkeypatch.setattr(logo, "draw_mark", _never)
    from app.agent.skills.builtins.app_html import vision
    monkeypatch.setattr(vision, "can_call_model", lambda: True)

    _svg, source = _run(logo.ensure_icon("plain", title="Plain", user_id="u"))
    assert source == "fallback"
    assert not drew


def test_a_refused_drawing_is_redrawn_with_the_reason(apps_dir, monkeypatch):
    """The refusal is written for this reader, so it is handed back verbatim.
    Measured: two attempts left one app in six on a holding mark, and the
    refusals were single fixable faults."""
    seen = []

    async def _ask(system, user, **_k):
        seen.append(user)
        if len(seen) == 1:
            return GOOD.replace("#C4703A", "#FF00FF")     # a stray colour
        return GOOD

    monkeypatch.setattr(logo, "_ask", _ask)
    svg = _run(logo.draw_mark(user_id="u", title="Snake", scene="A snake.",
                              palette=["#2F6B3A", "#F7F4EC", "#C4703A"]))
    assert svg == GOOD
    assert len(seen) == 2
    assert "REJECTED" in seen[1]
    assert "#ff00ff" in seen[1].lower()


def test_a_drawing_that_never_complies_falls_back_rather_than_shipping(apps_dir, monkeypatch):
    async def _ask(system, user, **_k):
        return GOOD.replace("#C4703A", "#FF00FF")

    monkeypatch.setattr(logo, "_ask", _ask)
    assert _run(logo.draw_mark(user_id="u", title="S", scene="A snake.",
                               palette=["#2F6B3A", "#F7F4EC"])) is None


# ── Round 25 follow-up: three holes an audit found in the geometry rule ──
#
# The measurer was right about geometry and wrong about what counts AS the
# mark. All three of these passed every composition rule while being obviously
# not an app icon, and the fourth was the opposite — a legitimate mark refused
# for something that is never drawn.

_P = ["#111111", "#4488ff", "#88ccff"]
_G = '<rect width="96" height="96" fill="#111111"/>'
_GLYPH = ('<circle cx="48" cy="48" r="26" fill="#4488ff"/>'
          '<rect x="34" y="34" width="28" height="28" fill="#88ccff"/>'
          '<path d="M30 62 L48 30 L66 62 Z" fill="#4488ff"/>')
_SPECK = ('<circle cx="48" cy="48" r="3" fill="#4488ff"/>'
          '<rect x="46" y="46" width="4" height="4" fill="#88ccff"/>'
          '<path d="M46 50 L48 46 L50 50 Z" fill="#4488ff"/>')


def _svg(inner: str) -> str:
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96" '
            f'width="96" height="96">{inner}</svg>')


def test_an_invisible_spacer_cannot_size_a_speck_into_a_glyph():
    """`fill="none"` with no stroke paints nothing, but it has geometry — so
    it measured like a shape. A 68x68 spacer plus a 3-unit dot satisfied the
    safe-area, size AND centring rules at once, while being a dot."""
    with pytest.raises(IconError):
        logo.sanitize_svg(
            _svg(_G + '<rect x="14" y="14" width="68" height="68" fill="none"/>'
                 + _SPECK), _P)


def test_specks_in_opposite_corners_are_not_a_centred_mark():
    """Every size rule measured the UNION of the shapes, and a union is not a
    mark: two 2-unit specks at opposite corners of the safe box measure 68x68,
    dead centre. The largest single shape has to carry the mark."""
    with pytest.raises(IconError) as exc:
        logo.sanitize_svg(_svg(
            _G + '<circle cx="16" cy="16" r="2" fill="#4488ff"/>'
            '<circle cx="80" cy="80" r="2" fill="#88ccff"/>'
            '<rect x="47" y="47" width="2" height="2" fill="#4488ff"/>'), _P)
    assert "biggest shape" in str(exc.value)


def test_a_definition_is_not_a_drawing():
    """Shapes inside `<defs>`/`<clipPath>` are never painted where they are
    written. Measuring them refused a perfectly centred mark because of a
    4-unit rect in a clip path — a FALSE refusal, which is the expensive kind:
    three of those and the app falls back to the plain holding bands."""
    logo.sanitize_svg(_svg(
        '<defs><rect x="0" y="0" width="4" height="4" fill="#4488ff"/></defs>'
        + _G + _GLYPH), _P)


def test_a_ground_that_exists_only_inside_a_mask_is_not_a_ground():
    """The same blindness in the other direction: a full-frame rect inside a
    `<mask>` satisfied the full-bleed-ground check for a drawing with no
    ground at all."""
    with pytest.raises(IconError) as exc:
        logo.sanitize_svg(_svg(
            '<mask id="m"><rect width="96" height="96" fill="#fff"/></mask>'
            + _GLYPH), _P)
    assert "full-bleed ground" in str(exc.value)


def test_use_is_refused_because_it_draws_where_nothing_measured():
    """`<use href="#id" x=... y=...>` re-draws a definition at an offset of its
    own, so the geometry scan measures it where it was WRITTEN rather than
    where it lands — the one way left to place a glyph the validator cannot
    see."""
    with pytest.raises(IconError):
        logo.sanitize_svg(
            _svg(_G + _GLYPH + '<use href="#x" x="40" y="40"/>'), _P)


def test_none_of_this_refuses_an_ordinary_good_mark():
    """The check that matters most: an over-strict validator is worse than a
    loose one, because `draw_mark` gets MAX_DRAW_ATTEMPTS tries and then the
    app degrades to the holding bands."""
    logo.sanitize_svg(_svg(_G + _GLYPH), _P)
    # A knockout crescent — a shape in the ground's own colour, which is how a
    # bite or a gap is drawn, and which legitimately reaches past the mark.
    logo.sanitize_svg(_svg(
        _G + '<circle cx="48" cy="48" r="28" fill="#4488ff"/>'
        '<circle cx="58" cy="40" r="22" fill="#111111"/>'
        '<rect x="30" y="66" width="36" height="10" fill="#88ccff"/>'), _P)
    # A gradient, defined in `<defs>` and referenced by the glyph.
    logo.sanitize_svg(_svg(
        '<defs><linearGradient id="g"><stop stop-color="#4488ff"/>'
        '<stop offset="1" stop-color="#88ccff"/></linearGradient></defs>'
        + _G + '<circle cx="48" cy="48" r="27" fill="url(#g)"/>'
        '<rect x="36" y="36" width="24" height="24" fill="#88ccff"/>'
        '<path d="M32 64 L48 32 L64 64 Z" fill="#4488ff"/>'), _P)
