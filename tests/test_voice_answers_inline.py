"""Voice answers in the turn; it does not file a job and walk away.

THE INCIDENT (founder's account, 2026-08-01T00:25Z, all rows real)

He asked in Farsi for University of Toronto professors working on LLMs. In
145 seconds the agent produced three background jobs and zero spoken answers:

    00:25:24  create_job → "Find UofT LLM professors"   cancelled  turn_interrupted
    00:25:57  spawn      → "UofT LLM/NLP professor …"   failed     infra_interrupted
    00:27:23  spawn      → "UofT LLM/NLP professor …"   failed     infra_interrupted

Both subagent rows recorded ``total_tokens=0`` and ``credit_spent=0.0`` across
the 19 minutes they were alive — they never executed at all. Meanwhile the
voice model told him «یه گزارش جمع‌وجور و به‌دردبخور به فارسی برات میاد»
("a compact, useful report in Farsi is coming to you"). Nothing came, so he
re-asked, and each re-ask minted another job.

The model was not misbehaving. It was following the FULL profile's decision
rules, which say verbatim "research … → call `create_job` FIRST" and route
"find <X> for me" to `browser` (a headless browser at tens of seconds a step)
rather than `web_search`. Voice inherited the web-chat prompt wholesale; its
only channel guidance was about tone.

THE FIX, in two layers — the same split ``SUBAGENT_DISABLED_TOOLS`` already
documents, because prompts are advisory and tool-list omission is hard:

  hard   ``VOICE_DISABLED_TOOLS`` removes create_job / update_job / spawn.
  soft   the prompt stops naming tools voice does not have, and states the
         answer-now contract.

Both layers are asserted here. The prompt assertions evaluate the REAL
expression out of agent_runner.py via ast — the same technique
test_channel_formatting_rules.py uses — so they run against the exact bytes
that reach the model, without importing the module or building a runner.

Run:
  pytest backend/tests/test_voice_answers_inline.py -v
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_BACKEND = Path(__file__).resolve().parent.parent
_AGENT_RUNNER = _BACKEND / "app" / "agent" / "agent_runner.py"


# ──────────────────────────────────────────────────────────────
# Snapshot renderer
# ──────────────────────────────────────────────────────────────

def _assign_value(target_desc: str) -> ast.expr:
    """The RHS expression of ``section_parts["platform_knowledge"] = (...)``.

    Takes the FIRST such assignment: a later one replaces the whole block with
    the diet variant under a flag, which is not what this test is about.
    """
    tree = ast.parse(_AGENT_RUNNER.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        t = node.targets[0]
        if (
            isinstance(t, ast.Subscript)
            and isinstance(t.value, ast.Name)
            and t.value.id == "section_parts"
            and isinstance(t.slice, ast.Constant)
            and t.slice.value == target_desc
        ):
            return node.value
    raise AssertionError(f'section_parts["{target_desc}"] assignment not found')


def _render_platform_knowledge(voice: bool) -> str:
    """Evaluate the real platform_knowledge expression for one channel."""
    expr = _assign_value("platform_knowledge")
    code = compile(ast.Expression(body=expr), "<pk>", "eval")
    return eval(code, {"__builtins__": {}}, {"_voice_now": voice})  # noqa: S307


def _channel_guidance_for(channel: str) -> str:
    """The real guidance string for one channel.

    ``_channel_guidance`` is not a dict — it is ``<table>.get(_channel_safe,
    <fallback>)``, i.e. already resolved at the assignment. Evaluating the
    whole Call rather than digging the Dict out of it means the lookup and
    its fallback are exercised too, which is what the model actually sees.

    The table itself moved to module level as ``CHANNEL_GUIDANCE`` (G-19b,
    so policy tests can reference it by name), so it is supplied to the
    eval namespace here. Evaluating the assignment expression — rather
    than calling the dict directly — is deliberate: it keeps the fallback
    branch inside what this helper covers.
    """
    from app.agent.agent_runner import CHANNEL_GUIDANCE

    tree = ast.parse(_AGENT_RUNNER.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "_channel_guidance"
        ):
            code = compile(ast.Expression(body=node.value), "<cg>", "eval")
            return eval(  # noqa: S307
                code,
                {"__builtins__": {}},
                {"_channel_safe": channel, "CHANNEL_GUIDANCE": CHANNEL_GUIDANCE},
            )
    raise AssertionError("_channel_guidance assignment not found")


@pytest.fixture(scope="module")
def pk_voice() -> str:
    return _render_platform_knowledge(voice=True)


@pytest.fixture(scope="module")
def pk_web() -> str:
    return _render_platform_knowledge(voice=False)


# ──────────────────────────────────────────────────────────────
# Layer 1 — the hard one: the tools are gone
# ──────────────────────────────────────────────────────────────

class TestVoiceDisabledTools:
    def test_voice_loses_the_three_deferral_tools(self):
        from app.agent.prompt_profile import disabled_tools_for_channel

        assert disabled_tools_for_channel("voice") == {
            "create_job", "update_job", "spawn",
        }

    def test_start_mission_survives(self):
        """The one deferral the user asks for in words keeps working."""
        from app.agent.prompt_profile import disabled_tools_for_channel

        assert "start_mission" not in disabled_tools_for_channel("voice")

    @pytest.mark.parametrize("channel", ["web", "app", "mobile", "telegram", None, ""])
    def test_no_other_channel_is_touched(self, channel):
        from app.agent.prompt_profile import disabled_tools_for_channel

        assert disabled_tools_for_channel(channel) == frozenset()

    @pytest.mark.parametrize("raw", ["VOICE", " voice ", "Voice"])
    def test_channel_match_is_normalised(self, raw):
        """resolve_channel is not guaranteed to have run on this value."""
        from app.agent.prompt_profile import disabled_tools_for_channel

        assert "spawn" in disabled_tools_for_channel(raw)


# ──────────────────────────────────────────────────────────────
# Layer 2 — the prompt stops naming tools voice cannot call
# ──────────────────────────────────────────────────────────────

class TestVoicePromptOmitsDeferralTools:
    @pytest.mark.parametrize("tool", ["create_job", "update_job", "spawn"])
    def test_absent_from_voice_prompt(self, pk_voice, tool):
        """A rule naming an absent tool is worse than no rule — it reliably
        produces "I can't do that from here" instead of the thing the model
        CAN do. This is the same failure mode as a tool missing from
        _REALTIME_NATIVE."""
        assert tool not in pk_voice

    @pytest.mark.parametrize("tool", ["create_job", "update_job"])
    def test_still_present_for_web(self, pk_web, tool):
        """The web/app path is untouched — this fix is surface-scoped."""
        assert tool in pk_web

    def test_web_keeps_the_create_job_first_rule(self, pk_web):
        # Round 4 (item 7): the rule still opens the job at the start of the
        # work — but IN THE SAME RESPONSE as the first tool calls, never as
        # a response by itself (that was one full LLM round-trip per turn).
        assert "call `create_job` in the SAME response as the first step's tool calls" in pk_web
        assert "never call update_job to mark completed" in pk_web

    def test_start_mission_is_offered_on_both(self, pk_voice, pk_web):
        assert "start_mission" in pk_voice
        assert "start_mission" in pk_web


class TestVoicePromptStatesTheAnswerNowContract:
    def test_search_is_the_named_tool(self, pk_voice):
        assert "web_search" in pk_voice

    def test_forbids_promising_a_later_deliverable(self, pk_voice):
        """The exact thing the agent did: promised a report, delivered none."""
        low = pk_voice.lower()
        assert "never promise" in low or "never promise a report" in low

    def test_says_the_work_happens_in_this_turn(self, pk_voice):
        assert "in this turn" in pk_voice.lower()


class TestBrowserIsNoLongerTheSearchTool:
    """'find <X> for me' routed to `browser` — a real headless browser at tens
    of seconds per step — on EVERY channel, voice included. Search answers a
    question; the browser is for pages you must operate."""

    def test_find_x_routes_to_web_search(self, pk_web):
        line = next(
            (l for l in pk_web.splitlines() if "'find <X> for me'" in l), None
        )
        assert line is not None, "the find-X decision rule vanished"
        assert "`web_search`" in line

    def test_browser_is_reserved_for_operating_a_page(self, pk_web):
        line = next(
            (l for l in pk_web.splitlines() if "'find <X> for me'" in l), None
        )
        assert "`browser` is for pages you must OPERATE" in line

    def test_the_rule_reaches_voice_too(self, pk_voice):
        assert "`web_search`" in pk_voice


# ──────────────────────────────────────────────────────────────
# Channel guidance: voice gets a behaviour contract, not just a tone note
# ──────────────────────────────────────────────────────────────

class TestVoiceChannelGuidance:
    def test_voice_entry_exists(self):
        assert _channel_guidance_for("voice").startswith("User is on the Toup voice")

    def test_it_is_more_than_a_tone_note(self):
        """Before the fix this was one sentence about markdown. `extension`
        already proved a behavioural block belongs here."""
        g = _channel_guidance_for("voice")
        assert len(g) > 400, "voice guidance is still tone-only"
        assert "BEHAVIOR" in g

    def test_it_names_search_and_forbids_deferral(self):
        g = _channel_guidance_for("voice")
        assert "`web_search`" in g
        assert "NEVER promise a deliverable" in g

    def test_it_handles_the_repeat_ask(self):
        """He re-asked twice and got a new job each time."""
        g = _channel_guidance_for("voice").lower()
        assert "repeats a request" in g

    def test_tone_rules_survived(self):
        g = _channel_guidance_for("voice")
        assert "No markdown" in g
        assert "spoken aloud" in g

    @pytest.mark.parametrize("other", ["web", "app", "mobile", "telegram"])
    def test_other_channels_unchanged_in_shape(self, other):
        """Nothing here should have leaked into the neighbours."""
        g = _channel_guidance_for(other)
        assert "NEVER promise a deliverable" not in g


# ──────────────────────────────────────────────────────────────
# The diet path — where this fix nearly died
#
# `PROMPT_DIET` swaps the entire platform_knowledge block for a compact
# literal. That literal had its OWN decision rules, and they said
# "multi-step work you'll finish THIS turn → `create_job` first" and routed
# search to `browser`. So flipping one env var would have reverted the whole
# prompt layer — and with VOICE_DISABLED_TOOLS still in force the result is
# strictly worse than before the fix: the model is told to call a tool that
# is not in its list, which is the "I can't do that from here" failure.
#
# The flag is off in production today. That is exactly why this is pinned:
# nothing else would catch it the day someone turns it on.
# ──────────────────────────────────────────────────────────────

class TestPromptDietIsVoiceAware:
    @pytest.mark.parametrize("tool", ["create_job", "update_job"])
    def test_voice_diet_names_no_deferral_tool(self, tool):
        from app.agent.prompt_diet import platform_knowledge_diet

        assert tool not in platform_knowledge_diet(voice=True)

    @pytest.mark.parametrize("tool", ["create_job", "update_job"])
    def test_web_diet_keeps_them(self, tool):
        from app.agent.prompt_diet import platform_knowledge_diet

        assert tool in platform_knowledge_diet(voice=False)

    def test_voice_diet_routes_search_to_web_search(self):
        from app.agent.prompt_diet import platform_knowledge_diet

        assert "`web_search`" in platform_knowledge_diet(voice=True)

    def test_voice_diet_forbids_the_promise(self):
        from app.agent.prompt_diet import platform_knowledge_diet

        assert "Never promise" in platform_knowledge_diet(voice=True)

    def test_back_compat_constant_is_the_web_variant(self):
        """test_prompt_diet.py pins this constant; it must stay byte-identical."""
        from app.agent.prompt_diet import (
            PLATFORM_KNOWLEDGE_DIET, platform_knowledge_diet,
        )

        assert PLATFORM_KNOWLEDGE_DIET == platform_knowledge_diet(False)

    def test_the_call_site_passes_the_channel(self):
        """A voice-aware builder wired up with a hardcoded False would pass
        every test above and still ship the bug."""
        src = _AGENT_RUNNER.read_text()
        assert "_platform_knowledge_diet(_voice_now)" in src


class TestVoiceDetectionCannotDriftFromChannelSafe:
    """`_voice_now` reads the raw `channel`; `_channel_safe` is
    resolve_channel(explicit=channel). Those must agree, or the prompt would
    take the voice branch while the guidance took the web one (or vice versa).
    resolve_channel only strips and lowercases the explicit value — no alias
    mapping — so they do. Pinned because an alias table added later would
    break it silently."""

    @pytest.mark.parametrize(
        "raw", ["voice", "Voice", " voice ", "VOICE", "web", "mobile", None, ""],
    )
    def test_raw_check_matches_resolved(self, raw):
        from app.agent.channel_util import resolve_channel

        voice_now = (raw or "").strip().lower() == "voice"
        channel_safe = resolve_channel(explicit=raw, site="test")
        assert voice_now == (channel_safe == "voice")

    def test_resolve_channel_is_still_called_with_explicit_only(self):
        """If a payload/conversation hint is added to that call, the raw check
        above stops being equivalent and both sites must move together."""
        src = _AGENT_RUNNER.read_text()
        i = src.index("_channel_safe = resolve_channel(")
        call = src[i:src.index(")", i)]
        assert "payload_hint" not in call
        assert "conversation_hint" not in call
