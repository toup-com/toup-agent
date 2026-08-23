"""Round 24 — the icon step never claims a mark it does not have.

The recorded regression: every recent app card showed the letter monogram —
the client's rendering for NO icon at all — while every build card said
"drew a fresh mark in the app's colours". The seam was `ensure_icon`: a
successful draw whose `_store_icon` failed was logged and returned as
``"model"`` anyway, so the step reported a mark that was not on disk, and a
fleet where every icon degraded to the bands was invisible in the logs.

The contract now (module docstring of :mod:`logo`): ``"model"`` means drawn
AND persisted; ``"fallback"`` means bands on disk; a raise means the step
must not claim a mark. Each test here is one clause of that.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest

from app.agent.skills.builtins.app_html import logo, store, vision
from app.agent.skills.builtins.app_html.store import AppStoreError

# Redrawn for round 25's art direction: absolute path commands, and the
# subject inside the 14–82 safe area rather than clipped by the frame.
GOOD = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96" width="96" '
    'height="96"><rect width="96" height="96" fill="#2B2724"/>'
    '<path d="M20 62 Q34 40 48 54 Q62 68 76 46" stroke="#F3EDE4" '
    'stroke-width="10" fill="none" stroke-linecap="round"/>'
    '<circle cx="72" cy="42" r="7" fill="#C1443A"/></svg>'
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


def _model_reachable(monkeypatch):
    monkeypatch.setenv("TOUP_APP_MODEL_CALLS", "1")
    monkeypatch.setattr(vision, "can_call_model", lambda: True)


def _write_app_with_palette(slug: str, title: str) -> None:
    store.write_app(slug, title,
                    '<!doctype html><html><head><style>:root{--bg:#2B2724;'
                    '--ink:#C1443A;--paper:#F3EDE4}</style></head><body><p>'
                    + "x" * 500 + "</p></body></html>")


def _drawing_succeeds(monkeypatch):
    async def _subject_ok(**_k):
        return "snake body", "The snake's own body turning a corner."

    async def _draw_ok(**_k):
        return GOOD

    monkeypatch.setattr(logo, "choose_subject", _subject_ok)
    monkeypatch.setattr(logo, "draw_mark", _draw_ok)


def test_a_mark_that_cannot_be_stored_is_never_claimed(apps_dir, monkeypatch):
    """Draw succeeds, disk refuses twice: `ensure_icon` RAISES — never
    ``("model")`` — so skill.py's except path emits the honest skipped step
    instead of "drew a fresh mark" over an empty directory."""
    _write_app_with_palette("timer", "Pomodoro Timer")
    _model_reachable(monkeypatch)
    _drawing_succeeds(monkeypatch)

    attempts = []

    def _refuse(*_a, **_k):
        attempts.append(1)
        raise OSError("disk says no")

    monkeypatch.setattr(logo, "_store_icon", _refuse)

    with pytest.raises(AppStoreError):
        _run(logo.ensure_icon("timer", title="Pomodoro Timer", user_id="u"))
    assert len(attempts) == 2          # the one retry happened
    assert not Path(logo.icon_path("timer")).exists()
    assert not Path(logo.sidecar_path("timer")).exists()


def test_one_store_hiccup_is_absorbed_by_the_retry(apps_dir, monkeypatch):
    """A single transient failure costs nothing: the retry lands, both files
    exist, and only then is ``"model"`` claimed."""
    _write_app_with_palette("timer", "Pomodoro Timer")
    _model_reachable(monkeypatch)
    _drawing_succeeds(monkeypatch)

    real_store = logo._store_icon
    attempts = []

    def _flaky(*a, **k):
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("transient")
        return real_store(*a, **k)

    monkeypatch.setattr(logo, "_store_icon", _flaky)

    svg, source = _run(logo.ensure_icon("timer", title="Pomodoro Timer",
                                        user_id="u"))
    assert (svg, source) == (GOOD, "model")
    assert len(attempts) == 2
    assert Path(logo.icon_path("timer")).is_file()
    assert Path(logo.sidecar_path("timer")).is_file()
    assert logo.read_sidecar("timer")["source"] == "model"


def test_no_subject_after_a_retry_degrades_loudly(apps_dir, monkeypatch, caplog):
    """`choose_subject` failing gets exactly one more sample; still nothing
    means the bands go to disk as ``"fallback"`` — and the degrade is a
    WARNING naming its branch, because a fleet where every icon degrades was
    silent."""
    _write_app_with_palette("timer", "Pomodoro Timer")
    _model_reachable(monkeypatch)

    asked = []

    async def _no_subject(**_k):
        asked.append(1)
        return "", ""

    async def _never(**_k):
        raise AssertionError("no subject must mean no drawing")

    monkeypatch.setattr(logo, "choose_subject", _no_subject)
    monkeypatch.setattr(logo, "draw_mark", _never)

    with caplog.at_level(logging.WARNING, logger=logo.logger.name):
        svg, source = _run(logo.ensure_icon("timer", title="Pomodoro Timer",
                                            user_id="u"))
    assert source == "fallback"
    assert len(asked) == 2             # one retry, not a loop
    assert Path(logo.icon_path("timer")).is_file()
    assert logo.read_sidecar("timer")["source"] == "fallback"
    assert "degraded to the holding bands" in caplog.text
    assert "no usable subject" in caplog.text


def test_a_refused_drawing_degrades_loudly(apps_dir, monkeypatch, caplog):
    """All draw attempts refused is the other degrade branch, and it names
    itself in the log too. draw_mark's own 3-attempt loop is the retry — no
    second loop around it."""
    _write_app_with_palette("timer", "Pomodoro Timer")
    _model_reachable(monkeypatch)

    drew = []

    async def _subject_ok(**_k):
        return "snake body", "The snake's own body turning a corner."

    async def _refused(**_k):
        drew.append(1)
        return None

    monkeypatch.setattr(logo, "choose_subject", _subject_ok)
    monkeypatch.setattr(logo, "draw_mark", _refused)

    with caplog.at_level(logging.WARNING, logger=logo.logger.name):
        _svg, source = _run(logo.ensure_icon("timer", title="Pomodoro Timer",
                                             user_id="u"))
    assert source == "fallback"
    assert len(drew) == 1
    assert "degraded to the holding bands" in caplog.text
    assert "draw refused" in caplog.text
