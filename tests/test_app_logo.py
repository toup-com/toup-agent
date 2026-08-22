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


def test_the_fallback_is_deterministic(apps_dir):
    """Same slug, same tile — across containers, restarts and test runs. An
    app whose colour changes between two opens does not look like one app."""
    a = logo.fallback_icon("snake", "Nokia Snake Classic")
    b = logo.fallback_icon("snake", "Nokia Snake Classic")
    c = logo.fallback_icon("pomodoro", "Pomodoro")
    assert a == b
    assert a != c


def test_the_monogram_reads_the_name_not_the_slug(apps_dir):
    assert ">NS<" in logo.fallback_icon("nokia-snake", "Nokia Snake")
    assert ">PO<" in logo.fallback_icon("pomodoro", "Pomodoro")
    # With no title it falls back to the slug, whose hyphen is a word break:
    # "BT", not "BU". A monogram of the first two letters of a kebab slug
    # ("BU", "NO", "PO") reads as a truncation rather than as initials.
    assert logo.initials("", "budget-tracker") == "BT"
    assert logo.initials("", "pomodoro") == "PO"


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


def test_the_icon_model_is_pinned():
    assert logo.LOGO_MODEL and logo.LOGO_MODEL != "None"


def test_a_slug_cannot_escape_the_icon_directory(apps_dir):
    from app.agent.skills.builtins.app_html.store import AppStoreError

    for bad in ("../outside", "a/b", ".."):
        with pytest.raises(AppStoreError):
            logo.icon_path(bad)
