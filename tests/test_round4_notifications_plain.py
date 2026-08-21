"""Round 4 (2026-08-19) — item 4 (no raw markdown on notification surfaces)
and item 3 (richer Live Activity / Dynamic Island payloads).

"**Gemini 3.7 Flash**" reached the lock screen with literal asterisks because
nothing between the model and APNs stripped markdown. One stripper
(`app.services.plain_text`) now runs at the platform choke points
(`apns_push` content-state + alert, `expo_push.build_message`) AND at every
agent-side producer BEFORE the preview is sliced (a `**` pair split by
`[:120]` would otherwise survive).
"""

from __future__ import annotations

import inspect
from pathlib import Path

from app.services.plain_text import plain_preview, strip_markdown


BACKEND = Path(__file__).resolve().parent.parent


# ── the stripper ────────────────────────────────────────────────────────

def test_strip_markdown_removes_syntax_and_keeps_words():
    assert strip_markdown("**Gemini 3.7 Flash** is out") == "Gemini 3.7 Flash is out"
    assert strip_markdown("see [the blog](https://blog.google/x) now") == "see the blog now"
    assert strip_markdown("use `gpt-5.6-terra` *today* _ok_ ~~old~~") == "use gpt-5.6-terra today ok old"
    assert strip_markdown("## Summary\n\n- **a**: b\n1. c\n> q") == "Summary\n\na: b\nc\nq"
    assert strip_markdown("```py\nprint(1)\n```\nDone [[navigate:/jobs]] [[Play X on Netflix]]") == "print(1)\n\nDone"


def test_a_chip_leaves_no_hole_behind():
    """Round 15. Cutting a chip out with a blunt `sub` shipped a real user
    "You can watch it in ." — in chat, and this surface (push bodies, Live
    Activity strings, SMS) strips chips the same way, so it would have sent
    the same sentence to their lock screen.

    Mirrors `frontend/src/modules/chat/chipDirectives.ts`; the two must agree,
    because the same message goes out on both."""
    # The clause whose object was the chip goes with it.
    assert strip_markdown("You can watch it in [[navigate:/jobs]].") == "You can watch it."
    assert strip_markdown("Watch it at [[navigate:/jobs]]") == "Watch it."
    # A line that was only a chip leaves nothing, not a stray full stop.
    assert strip_markdown("[[navigate:/jobs]].") == ""
    # The gap mid-sentence closes up.
    assert strip_markdown("Done! [[Looks great!]]") == "Done!"
    assert strip_markdown("It is on. [[Change it]] and tell me") == "It is on. and tell me"
    # Ordinary prose ending in the same word is untouched — no chip was cut.
    assert strip_markdown("The toggle is on.") == "The toggle is on."
    assert strip_markdown("Everyone is here") == "Everyone is here"
    assert strip_markdown("| Model | Score |\n|---|---|\n| Fable 5 | 91 |") == "Model Score\n\nFable 5 91"
    assert strip_markdown("<b>bold</b><br>x") == "boldx"


def test_strip_markdown_leaves_prose_alone():
    for s in ("Price: $5 and $10 today", "a * b * c", "2*3*4", "snake_case_name stays",
              "a < b and <3", "5 - 3 = 2", "Q3 2026 - review", "https://x.example/a_b_c"):
        assert strip_markdown(s) == s, s
    # escapes become the literal char; LaTeX-looking $…$ becomes its body
    assert strip_markdown("\\*not bold\\* and $x^2$") == "*not bold* and x^2"


def test_strip_markdown_is_total_and_idempotent():
    assert strip_markdown(None) == "" and strip_markdown("") == ""
    s = "**A** [b](http://c) `d`\n\n- e"
    once = strip_markdown(s)
    assert strip_markdown(once) == once
    assert plain_preview("**Truncated bold at the end**", 12) == "Truncated bo"


def test_plain_preview_strips_before_slicing():
    """The order matters: slicing first would leave a lone '**'."""
    txt = "**" + "x" * 118 + "** tail"
    assert "*" not in plain_preview(txt, 120)
    assert plain_preview("line one\nline **two**", 100) == "line one line two"


# ── platform choke points ───────────────────────────────────────────────

def test_apns_content_state_and_alert_are_plain():
    from app.services import apns_push as P
    st = P._content_state(
        "**Bold title**", "*sub* [x](http://y)", 0.5,
        extra={"stepName": "Read the **top** results", "preview": "**Gemini 3.7 Flash** wins",
               "jobType": "search", "chatId": "abc", "messageId": "def",
               "stepsDone": 1, "stepsTotal": 3, "percent": 50},
    )
    assert st["title"] == "Bold title" and st["subtitle"] == "sub x"
    assert st["stepName"] == "Read the top results"
    assert st["preview"] == "Gemini 3.7 Flash wins"
    assert st["chatId"] == "abc" and st["jobType"] == "search"
    assert st["stepsDone"] == 1 and st["stepsTotal"] == 3
    al = P._alert("✅ Done: **Compare models**", "**Fable 5** leads — see [x](http://y)\n- item")
    assert al["title"] == "✅ Done: Compare models"
    assert al["body"] == "Fable 5 leads — see x\nitem"


def test_expo_message_is_plain():
    from app.services.expo_push import build_message
    m = build_message("ExponentPushToken[x]", "**Answer ready**", "**Yes** — [docs](http://d)", {}, "high")
    assert m["title"] == "Answer ready" and m["body"] == "Yes — docs"


# ── agent-side producers strip BEFORE they slice ────────────────────────

def test_producers_strip_before_slicing():
    ws = (BACKEND / "app" / "api" / "ws_chat.py").read_text()
    # 2026-08-19: the preview rides the same push-copy gate as the alert
    # body — answer_preview drops the agent's opening-ack narration before
    # slicing (strip-then-slice still holds inside answer_preview).
    assert '_answer_data["preview"] = _answer_preview(response.text, 120, fallback="") or None' in ws
    assert 'body=_ap(response.text, 180) or None' in ws
    ar = (BACKEND / "app" / "agent" / "agent_runner.py").read_text()
    assert '_preview = _answer_preview(final_text, 100, fallback="Finished.")' in ar
    so = (BACKEND / "app" / "agent" / "subagent_orchestrator.py").read_text()
    assert 'data["preview"] = _plain_preview(str(preview), 120)' in so
    te = (BACKEND / "app" / "agent" / "tool_executor.py").read_text()
    assert '"label": _plain(str(label), 120) or str(label)' in te   # step labels stored plain
    # 2026-08-19: the title is additionally gated for internal vocabulary
    # (humanize_label) — model-authored labels are lock-screen headlines.
    assert 'title=f"🛠 Working on: {_hl_cj(_plain(title, 150))}"' in te
    assert "_job_title_plain" in te
    ap = (BACKEND / "app" / "agent" / "routines" / "autopilot_handler.py").read_text()
    assert "_strip_md(" in ap


def test_telegram_fallback_lane_is_not_stripped():
    """Telegram/WhatsApp render markdown on purpose — the strip is per-lane."""
    nd = (BACKEND / "app" / "api" / "notify_deliver.py").read_text()
    assert "strip_markdown" not in nd and "plain_preview" not in nd


# ── item 3: the interim beacon carries job step context ─────────────────

def test_turn_progress_beacon_carries_step_context_after_step_change():
    import asyncio
    from app.agent import turn_progress as TP

    sent = []

    async def fake_notify(**kw):
        sent.append(kw)
        return "row"

    async def go():
        em = TP.TurnProgressEmitter(mission_id="chatturn:x", mission_title="t", base_progress=5, ceiling=90)
        import app.services.agent_notify_client as N
        orig = N.notify
        N.notify = fake_notify
        try:
            await em.on_step_change({"job_id": "J", "step_index": 1, "step_name": "Read the results",
                                     "steps_total": 3, "job_type": "compare"})
            await em.on_tool_start("web_fetch")
        finally:
            N.notify = orig
    asyncio.run(go())
    assert len(sent) == 1
    d = sent[0]["data"]
    assert d["step_name"] == "Read the results"
    assert d["job_type"] == "compare" and d["steps_total"] == 3 and d["steps_done"] == 1
    # discrete step progress wins over the interpolated curve when we have it
    assert d["progress"] >= 5 + int((90 - 5) * (1 / 3))


def test_step_change_payload_carries_job_type_for_the_icon():
    from app.agent.step_tracker import StepTracker
    import json
    st = StepTracker()
    st.observe("create_job", {"steps": ["a", "b"]}, json.dumps({"job_id": "J", "steps": 2, "job_type": "verify"}))
    assert st.event_fields()["job_type"] == "verify"
