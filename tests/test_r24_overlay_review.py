"""Round 24: the visual reviewer must know the screenshot is POST-press.

The recorded Ping-Pong shipped "checked" while its serve card never dismissed.
Two independent guards now catch that class — the render gate's own
interaction assertion (test_app_verification.py §8b) and the visual reviewer,
which used to be handed a post-press screenshot with a system prompt that
whitelisted a visible start screen as "a normal state". This pins the reviewer
half: given a `pressed_start` label, the ask tells the model the shot is after
the press, and the system prompt makes a surviving start overlay a defect.

Run: cd backend && pytest tests/test_r24_overlay_review.py -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("ENVIRONMENT", "development")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent.skills.builtins.app_html import vision  # noqa: E402


def test_system_prompt_conditions_the_start_screen_whitelist():
    sp = vision._SYSTEM
    # The blanket "a start screen is a normal state" is now conditional…
    assert "UNLESS you are told the start control was already pressed" in sp
    # …and a start screen surviving its own press is named a defect.
    assert "failing to start" in sp


def test_pressed_start_label_reaches_the_ask(monkeypatch):
    # Capture the messages handed to the model without a live model call.
    captured = {}

    async def _fake_call_system_llm(**kwargs):
        captured.update(kwargs)
        return '{"ok": true}'

    import app.services.internal_llm as il
    monkeypatch.setattr(il, "call_system_llm", _fake_call_system_llm)
    monkeypatch.setattr(vision, "enabled", lambda: True)
    monkeypatch.setattr(vision, "can_call_model", lambda: True)

    import asyncio
    look = asyncio.run(vision.review_screenshot(
        b"\x89PNG-not-a-real-image",
        user_id="u1",
        title="Pocket Pong",
        purpose="a tiny tennis game",
        pressed_start="Serve",
    ))
    assert look.ran is True
    text_blocks = [
        c["text"] for c in captured["messages"][0]["content"]
        if isinstance(c, dict) and c.get("type") == "text"
    ]
    joined = "\n".join(text_blocks)
    assert "Serve" in joined
    assert "pressed" in joined.lower()
    assert "should be the app RUNNING" in joined


def test_no_pressed_start_leaves_the_ask_unchanged(monkeypatch):
    captured = {}

    async def _fake_call_system_llm(**kwargs):
        captured.update(kwargs)
        return '{"ok": true}'

    import app.services.internal_llm as il
    monkeypatch.setattr(il, "call_system_llm", _fake_call_system_llm)
    monkeypatch.setattr(vision, "enabled", lambda: True)
    monkeypatch.setattr(vision, "can_call_model", lambda: True)

    import asyncio
    asyncio.run(vision.review_screenshot(
        b"\x89PNG", user_id="u1", title="Notes", purpose="jot things down",
    ))
    text_blocks = [
        c["text"] for c in captured["messages"][0]["content"]
        if isinstance(c, dict) and c.get("type") == "text"
    ]
    joined = "\n".join(text_blocks)
    assert "should be the app RUNNING" not in joined
