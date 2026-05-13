"""Ticket 4 regression test — sidebar scroll bleed.

Frontend bug: scrolling the chat pane dragged the sidebar's chat-
history list along, eventually scrolling it to its own top. Root
cause: both panes used `overflow-y-auto` but neither used
`overscroll-behavior: contain`, so scroll events bubbled to the body
and chained across panes.

Fix: add `overscroll-contain` (Tailwind utility) to both scroll
containers. This test pins both classes via source-grep so a future
className cleanup can't silently drop the guard.

Source-grep is used because the repo's frontend doesn't have a Vitest
suite wired up at the backend test runner; pinning at the file level
gives us regression coverage that runs in CI alongside the backend.
"""

from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parent.parent.parent
_HUB_PAGE = (REPO / "frontend/src/pages/HubPageV2.tsx").read_text()
_CHAT_PAGE = (REPO / "frontend/src/modules/chat/ChatPage.tsx").read_text()


def test_sidebar_scroll_container_has_overscroll_contain():
    """The sidebar's chat-history scroll container must carry
    `overscroll-contain` (Tailwind → overscroll-behavior: contain) so
    scroll events don't bubble out and drag adjacent panes."""
    # The exact substring of the className combo we shipped.
    assert "overflow-y-auto overscroll-contain" in _HUB_PAGE, (
        "HubPageV2.tsx's sidebar scroll container must use "
        "`overflow-y-auto overscroll-contain`. Without overscroll-"
        "contain, the chat pane's scroll events bubble through and "
        "drag the sidebar — the original Ticket 4 symptom."
    )


def test_chat_message_scroller_has_overscroll_contain():
    """The chat message pane's `<main>` scroller must also carry the
    containment so scrolls don't chain in the OTHER direction."""
    assert "overflow-y-auto overscroll-contain" in _CHAT_PAGE, (
        "ChatPage.tsx's <main> scroll container must use "
        "`overflow-y-auto overscroll-contain` so scrolling past the "
        "ends doesn't chain out to the sidebar."
    )
