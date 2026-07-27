"""
Tests for W1.8 — channel-aware formatting rules + scheduled_tasks dedupe.

(a) The "# ── 7. Formatting rules (channel-aware)" dispatch in agent_runner.py
    used to send EVERY non-(app/web/vibecoding) channel down a Telegram-worded
    branch that taught Telegram-only [[button:LABEL|CALLBACK_DATA]] and
    [[reaction:EMOJI]] syntax — including mobile (whose Runtime Context line
    explicitly says "NOT Telegram"), voice, extension, discord, slack and
    whatsapp, where those markers leak into the message body as literal text.
    Post-fix: the Telegram wording + button/reaction teaching is gated on
    channel == "telegram" only; every other non-web channel gets a neutral
    messaging-surface variant (plain-text discipline, no tables, no marker
    syntax). The web/app/vibecoding branch is untouched.

    The snapshot renderer below extracts the real dispatch from the source
    via ast and evaluates each branch's string expression per channel, so the
    assertions run against the exact bytes that land in the system prompt —
    without importing agent_runner (heavy deps) or spinning up an AgentRunner.

(b) scheduled_tasks.py defined run_retrieval_feedback_analysis twice
    (byte-identical bodies; the second silently shadowed the first). Post-fix
    exactly one definition remains and the agent_main.py import still resolves.

Run:
  pytest backend/tests/test_channel_formatting_rules.py -v
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_BACKEND = Path(__file__).resolve().parent.parent
_AGENT_RUNNER = _BACKEND / "app" / "agent" / "agent_runner.py"
_SCHEDULED_TASKS = _BACKEND / "app" / "scripts" / "scheduled_tasks.py"


# ──────────────────────────────────────────────────────────────
# Snapshot renderer: evaluate the real formatting dispatch per channel
# ──────────────────────────────────────────────────────────────

def _formatting_dispatch() -> ast.If:
    """Locate the `if _channel_safe in ("app", "web", "vibecoding"):` node
    that assigns section_parts["formatting"]."""
    tree = ast.parse(_AGENT_RUNNER.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        t = node.test
        if (
            isinstance(t, ast.Compare)
            and isinstance(t.left, ast.Name)
            and t.left.id == "_channel_safe"
            and len(t.ops) == 1
            and isinstance(t.ops[0], ast.In)
            and isinstance(t.comparators[0], ast.Tuple)
        ):
            elts = {getattr(e, "value", None) for e in t.comparators[0].elts}
            if elts == {"app", "web", "vibecoding"}:
                return node
    raise AssertionError("formatting-rules channel dispatch not found in agent_runner.py")


def _eval_expr(expr: ast.expr, channel: str):
    return eval(  # noqa: S307 — evaluating string literals from our own source
        compile(ast.Expression(body=expr), "<formatting-branch>", "eval"),
        {"_channel_safe": channel},
    )


def formatting_section(channel: str) -> str:
    """Render section_parts["formatting"] exactly as agent_runner would for
    the given resolved channel."""
    node = _formatting_dispatch()
    while True:
        if _eval_expr(node.test, channel):
            stmts = node.body
        elif len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
            node = node.orelse[0]
            continue
        else:
            stmts = node.orelse
        assign = next(s for s in stmts if isinstance(s, ast.Assign))
        return _eval_expr(assign.value, channel)


NEUTRAL_CHANNELS = (
    "voice", "extension", "discord", "slack", "whatsapp", "unknown",
)
TELEGRAM_DELIVERY_CHANNELS = ("telegram", "cron", "heartbeat")


@pytest.mark.parametrize("channel", TELEGRAM_DELIVERY_CHANNELS)
def test_telegram_delivery_channels_keep_buttons_and_reactions(channel):
    """cron and heartbeat turns are delivered to the user's Telegram by the
    bot, so they keep the Telegram-shaped rules, not the neutral variant."""
    fs = formatting_section(channel)
    assert "You are communicating via Telegram" in fs
    assert "[[button:LABEL|CALLBACK_DATA]]" in fs
    assert "[[reaction:EMOJI]]" in fs


def test_mobile_keeps_quick_reply_buttons_without_telegram_framing():
    """The native app renders [[button:Label|value]] as quick-reply chips
    (ChatMarkdown.tsx), so mobile keeps the button teaching — but is never
    told it's on Telegram and is not taught [[reaction (app doesn't render it)."""
    fs = formatting_section("mobile")
    assert "Telegram" not in fs
    assert "[[button:LABEL|CALLBACK_DATA]]" in fs
    assert "[[reaction" not in fs
    assert "LaTeX" in fs


@pytest.mark.parametrize("channel", NEUTRAL_CHANNELS)
def test_non_telegram_messaging_channels_get_neutral_variant(channel):
    """Mobile's runtime line says 'NOT Telegram'; voice has no text UI at all;
    extension/discord/slack/whatsapp render [[...]] markers as literal text.
    None of them may be told they're on Telegram or taught marker syntax."""
    fs = formatting_section(channel)
    assert "Telegram" not in fs, f"{channel} formatting still mentions Telegram"
    assert "[[button" not in fs, f"{channel} formatting teaches [[button syntax"
    assert "[[reaction" not in fs, f"{channel} formatting teaches [[reaction syntax"
    # Plain-text discipline survives the rewrite.
    assert "LaTeX" in fs
    assert "Do NOT use tables" in fs


@pytest.mark.parametrize("channel", NEUTRAL_CHANNELS)
def test_neutral_variant_is_identical_across_channels(channel):
    """The neutral branch is one static string — per-channel tone lives in the
    Runtime Context channel line, not here (prompt-cache friendly)."""
    assert formatting_section(channel) == formatting_section("voice")


def test_web_app_vibecoding_branch_untouched():
    web = formatting_section("web")
    assert "web browser" in web
    assert "# Action Buttons" in web
    assert "[[Label]]" in web
    app = formatting_section("app")
    assert "inside their app" in app
    # vibecoding shares the content-area branch.
    assert formatting_section("vibecoding") == web


def test_callback_buttons_only_on_rendering_surfaces():
    for channel in ("web", "app", "vibecoding") + NEUTRAL_CHANNELS:
        assert "[[button:" not in formatting_section(channel), channel


# ──────────────────────────────────────────────────────────────
# (b) scheduled_tasks.py: exactly one run_retrieval_feedback_analysis
# ──────────────────────────────────────────────────────────────

def test_single_run_retrieval_feedback_analysis_definition():
    tree = ast.parse(_SCHEDULED_TASKS.read_text())
    defs = [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name == "run_retrieval_feedback_analysis"
    ]
    assert len(defs) == 1, (
        f"expected exactly one run_retrieval_feedback_analysis definition, "
        f"found {len(defs)} at lines {[d.lineno for d in defs]}"
    )


def test_agent_main_import_still_resolves():
    """agent_main.py:1030 imports this symbol — keep it importable + async."""
    import asyncio
    from app.scripts.scheduled_tasks import run_retrieval_feedback_analysis
    assert asyncio.iscoroutinefunction(run_retrieval_feedback_analysis)
