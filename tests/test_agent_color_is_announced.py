"""The one colour write that runs on every page load never announced itself.

Found 2026-08-13. The sidebar orb rendered the user's agent colour (navy
#1D4ED8) while the "New chat" label beside it stayed purple. Two sources of
truth, disagreeing:

  - the orb reads `--agent-color` off documentElement, which App.tsx sets
    from the boot config fetch;
  - every ACCENT TOKEN (`--t-accent-ink`, used for that label) is derived by
    ThemeProvider from its own copy, which it updates ONLY from the
    `agent-color-changed` bus.

`api.ts` announces on sign-out. SoulPage announces on change. App.tsx's boot
fetch — the only one that runs on every load — wrote the property silently.

ThemeProvider had a safety net for exactly this, and it was dead code:

    setBusColor((prev) => prev ?? readVar())

`index.css` declares `--agent-color: #9B59B6` as a static floor, so
`readVar()` never returns null, so `prev` is never null, so the `??` always
short-circuits. A guard whose precondition was destroyed by a default in a
stylesheet — invisible to tsc, to the build, and to every other check.
"""

from __future__ import annotations

import pathlib
import re

import pytest

FRONTEND = pathlib.Path(__file__).resolve().parents[2] / "frontend" / "src"


def _read(rel: str) -> str:
    p = FRONTEND / rel
    if not p.is_file():
        pytest.skip(f"{rel} not present in this checkout")
    return p.read_text()


def test_the_boot_config_fetch_announces_the_colour():
    """The write that runs on every page load must reach the theme bus."""
    src = _read("App.tsx")
    i = src.index("setProperty('--agent-color'")
    window = src[i: i + 400]
    assert "agent-color-changed" in window, (
        "App.tsx sets --agent-color without dispatching agent-color-changed. "
        "ThemeProvider never learns, so every accent token stays on the CSS "
        "floor colour while the orb shows the real one."
    )


def test_clearing_the_colour_is_announced_too():
    """Otherwise the previous account's accent survives into the next one —
    the leak api.ts:71 already guards on sign-out."""
    src = _read("App.tsx")
    i = src.index("removeProperty('--agent-color')")
    window = src[i: i + 400]
    assert "agent-color-changed" in window, (
        "clearing the colour is silent; the old accent persists in the theme"
    )


def test_the_safety_reread_is_not_gated_on_the_value():
    """`prev ?? readVar()` can never fire while index.css declares a floor for
    --agent-color. Gate on whether the BUS was heard instead."""
    src = _read("shared/theme.tsx")
    i = src.index("requestAnimationFrame")
    window = src[i: i + 300]
    assert "prev ?? readVar()" not in window, (
        "the re-read is gated on the value being null, which the CSS floor "
        "guarantees it never is — this is dead code that reads as a safety net"
    )
    assert "heardBus" in window, (
        "the re-read must be gated on whether a real bus event arrived"
    )


def test_the_css_floor_still_exists():
    """The test above is only meaningful while the floor exists. If the floor
    is removed, `prev ?? readVar()` would start working and this reasoning
    must be revisited rather than silently inherited."""
    css = _read("index.css")
    assert re.search(r"--agent-color:\s*#[0-9a-fA-F]{6}", css), (
        "the --agent-color CSS floor is gone; re-derive whether the bus gate "
        "is still the right mechanism"
    )


def test_the_bus_handler_records_that_it_fired():
    """The gate is only correct if the handler sets the flag — otherwise the
    rAF re-read clobbers a real event that arrived in the same frame."""
    src = _read("shared/theme.tsx")
    start = src.index("function useAgentColorFromBus")
    end = src.index("return explicit ?? busColor", start)
    block = src[start:end]
    handler = block.index("const handler")
    raf = block.index("requestAnimationFrame")
    assert "heardBus.current = true" in block[handler:raf], (
        "the bus handler does not record that it fired, so the safety re-read "
        "can overwrite a colour that genuinely arrived on the bus"
    )
