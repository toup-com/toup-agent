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

GOOD = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96" width="96" '
    'height="96"><rect width="96" height="96" rx="22" fill="#2F6B3A"/>'
    '<path d="M24 60 q12 -20 24 0 t24 0" stroke="#F7F4EC" stroke-width="7" '
    'fill="none" stroke-linecap="round"/>'
    '<circle cx="70" cy="40" r="6" fill="#C4703A"/></svg>'
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
        '<circle cx="50" cy="50" r="30" fill="#F7F4EC"/>'
        '<path d="M10 80 L90 80 L50 20 Z" fill="none" stroke="#C4703A" '
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
