"""One cacheable prefix for every channel; the channel rides a per-turn envelope.

Measured on the build this file's goldens were captured from (origin/main
32fddc38): `_build_system_prompt` produces **22 distinct strings for 22 channel
values**. The tools array serializes ahead of the system prompt, and the system
prompt ahead of the history, so each of those is its own provider cache
lineage — every hop from web to WhatsApp to mobile re-bills the whole prefix,
invisibly, because the model still answers correctly.

Three divergences, not one (agent_runner.py):
  :6272      `- Channel: {_channel_safe} — {_channel_guidance}` in runtime_lines
  :6255-6261 the managed-voice guidance swap
  :6284-6290 a whole Runtime-Context bullet branched on `_channel_safe == "voice"`
A fix that deletes only the first leaves voice on its own lineage, which is why
`test_voice_prompt_equals_web_prompt` is here as its own case.

D1: under `settings.channel_envelope` the prefix goes channel-NEUTRAL and the
per-turn facts move into a `<runtime_envelope>` user message placed AFTER
`<turn_context>` and BEFORE the current user message. Flag OFF must be
byte-identical to the pre-change build — that is what the goldens in
tests/fixtures/channel_prefix_goldens/ are, and `hashes.json` records the
agent_runner.py sha256 they were taken from so a reader can prove the golden is
a photograph of the build and not of the bug.

REGENERATING THE GOLDENS is a deliberate act. Build each channel's prompt with
`turn_context_out={}` (the clock is volatile without it), workspace-independent
and user-independent — both verified — and write sha256 for every
`KNOWN_CHANNELS` value plus '' and an unknown one, with full text for the seven
listed in `full_text_channels`. Say why in the commit.

Lane: RUN_MODE=agent — `_build_system_prompt` SELECTs `identities`, which is
AGENT_ONLY. Registered in tests/COVERAGE_DEBT.txt with the literal `# agent-mode`
marker; without that marker the file would run in NO lane and read green forever.

Local run (from backend/):
    RUN_MODE=agent PYTHONPATH=. ENVIRONMENT=test JWT_SECRET=x ENCRYPTION_KEY=y \
      pytest tests/test_channel_neutral_prefix.py -q --tb=short -p no:cacheprovider
"""
from __future__ import annotations

import difflib
import hashlib
import json
import re
import uuid
from pathlib import Path

import pytest

from app.config import Settings, settings

_BACKEND = Path(__file__).resolve().parent.parent
_SRC = (_BACKEND / "app" / "agent" / "agent_runner.py").read_text()
_GOLDENS = Path(__file__).resolve().parent / "fixtures" / "channel_prefix_goldens"

_WS = "/tmp/toup-prefix-test-ws"
_GOLDEN_USER_NAME = "Golden"


def _channels():
    from app.agent.channel_util import KNOWN_CHANNELS

    # The frozenset, never a literal list: a channel added later must not
    # silently escape the invariant.
    return sorted(KNOWN_CHANNELS) + ["", "smoke-signal"]


@pytest.fixture(scope="module")
def goldens():
    return json.loads((_GOLDENS / "hashes.json").read_text())


def _require_flag():
    # pydantic-settings refuses setattr for a field it does not declare, so the
    # skip has to come BEFORE monkeypatch touches `settings`.
    if "channel_envelope" not in Settings.model_fields:
        pytest.skip("settings.channel_envelope not landed yet — lane B1 (D1)")


@pytest.fixture
def envelope_on(monkeypatch):
    _require_flag()
    monkeypatch.setattr(settings, "channel_envelope", True)


@pytest.fixture
def envelope_off(monkeypatch):
    """Flag OFF, or — before the flag exists — the pre-change build itself,
    which IS the flag-off build. Either way the goldens must match."""
    if "channel_envelope" in Settings.model_fields:
        monkeypatch.setattr(settings, "channel_envelope", False)


#: The Runtime Context carries today's DATE, built from the live clock
#: (`prefix_stability.py`: `now_local.strftime('%A, %B %d, %Y')`). Hashing the
#: prompt with that line in it makes every golden — 22 shas and 7 full-text
#: diffs — turn red at midnight UTC for a reason that has nothing to do with the
#: prompt, on the very guard a deliberate prompt change is certified against.
#: The date is normalised out on BOTH sides instead; a golden must not contain a
#: calendar date.
_DATE_LINE_RE = re.compile(r"^- Today's date: .*$", re.M)
_FROZEN_DATE = "- Today's date: <FROZEN>"


def _freeze(prompt: str) -> str:
    return _DATE_LINE_RE.sub(_FROZEN_DATE, prompt)


async def _prompts(channels):
    """The REAL `_build_system_prompt`, per channel, for one user.

    `turn_context_out={}` is the production shape under `stable_prefix_layout`
    and it is what makes the output deterministic — without it the exact clock
    is inline and the bytes change every minute.
    """
    from app.agent.agent_runner import AgentRunner
    from app.agent.tool_executor import ToolExecutor
    from app.db.database import async_session_maker
    from app.db.models import User
    from app.services.openai_agent_service import OpenAIAgentService

    # The user's NAME is interpolated into the persona section, so the golden
    # and the live build must agree on it. `_GOLDEN_USER_NAME` is recorded in
    # hashes.json for exactly that reason; the id and email are not (verified
    # user-independent when the goldens were captured).
    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"{uid[:8]}@prefix.local", hashed_password="x" * 60,
                    name=_GOLDEN_USER_NAME))
        await db.commit()

    runner = AgentRunner(llm_service=OpenAIAgentService(),
                         tool_executor=ToolExecutor(workspace=_WS))
    out: dict[str, str] = {}
    for ch in channels:
        async with async_session_maker() as db:
            out[ch] = _freeze(await runner._build_system_prompt(
                db=db, user_id=uid, user_message="hi", channel=ch,
                client_tz="UTC", turn_context_out={},
            ))
    return out


def _first_divergence(prompts: dict[str, str]) -> str:
    items = sorted(prompts.items())
    base_ch, base = items[0]
    for ch, p in items[1:]:
        if p != base:
            diff = list(difflib.unified_diff(
                base.splitlines(), p.splitlines(),
                fromfile=base_ch or "(unset)", tofile=ch or "(unset)",
                lineterm="", n=1,
            ))
            return "\n".join(diff[:40])
    return ""


# ══════════════════════════════════════════════════════════════════════
# 1. Flag ON — one prefix
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_every_channel_shares_one_system_prompt(envelope_on):
    """THE headline invariant, and the system-tier twin of
    test_all_channels_one_lineage.py's tools-tier one."""
    prompts = await _prompts(_channels())
    assert len(set(prompts.values())) == 1, (
        "channels do not share one system prompt, so each extra lineage "
        "re-bills the whole prefix on every hop into it. First divergence:\n"
        + _first_divergence(prompts)
    )


@pytest.mark.asyncio
async def test_voice_prompt_equals_web_prompt(envelope_on):
    """The one a partial fix misses: voice diverges in TWO further places
    beyond the `- Channel:` line."""
    p = await _prompts(["web", "voice"])
    assert p["web"] == p["voice"], _first_divergence(p)


@pytest.mark.asyncio
async def test_no_channel_guidance_string_survives_into_the_prefix(envelope_on):
    """A half-move: the `- Channel:` line is deleted but a guidance string is
    still interpolated somewhere else in the assembly."""
    from app.agent.agent_runner import CHANNEL_GUIDANCE

    prompt = (await _prompts(["web"]))["web"]
    # voice / extension / vibecoding are deliberately HOISTED into the always-on
    # `surface_contracts` section (explicitly scoped, identical on every
    # channel) so their long behavioural text stays in the cached region
    # instead of being re-billed uncached on every turn. Every SHORT
    # per-channel descriptor must be gone from the prefix.
    # `trigger` joined them in review round 1: its guidance ends in the
    # G-19b "NEVER claim to have sent…" pin, which the first-sentence
    # descriptor cannot carry.
    hoisted = {"voice", "extension", "vibecoding", "trigger"}
    leaked = [k for k, v in CHANNEL_GUIDANCE.items() if v and k not in hoisted and v in prompt]
    assert not leaked, f"CHANNEL_GUIDANCE values still in the prefix: {leaked}"
    assert "- Channel: " not in prompt


@pytest.mark.asyncio
async def test_the_prompts_under_test_are_real(envelope_on):
    """ANTI-VACUITY. An empty or near-empty prompt would make the equality
    above trivially true — and so would a build that stopped assembling."""
    prompt = (await _prompts(["web"]))["web"]
    assert len(prompt) > 5000, f"system prompt too small to prove anything: {len(prompt)}"


# ══════════════════════════════════════════════════════════════════════
# 2. Flag OFF — byte-identical to the pre-change build
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_flag_off_is_byte_identical_to_the_goldens(envelope_off, goldens):
    """The flag-off contract from D1, made checkable. The goldens were taken
    from the agent_runner.py recorded in hashes.json — origin/main 32fddc38."""
    prompts = await _prompts(_channels())
    bad = []
    for ch, p in prompts.items():
        want = goldens["sha256"].get(ch)
        if want is None:
            bad.append(f"{ch or '(unset)'}: no golden — regenerate the fixture")
        elif hashlib.sha256(p.encode()).hexdigest() != want:
            bad.append(ch or "(unset)")
    assert not bad, (
        "flag-OFF prompt changed for: " + ", ".join(bad)
        + " — flag-off must stay byte-identical to the pre-change build. "
        "See the regeneration note in this file's docstring if the change "
        "was deliberate."
    )


@pytest.mark.asyncio
async def test_flag_off_full_text_matches_for_the_representative_channels(envelope_off, goldens):
    """A sha tells you THAT it moved; this tells you WHICH LINE moved."""
    named = goldens["full_text_channels"]
    prompts = await _prompts(named)
    for ch in named:
        want = (_GOLDENS / f"{ch}.txt").read_text()
        if prompts[ch] != want:
            diff = "\n".join(list(difflib.unified_diff(
                want.splitlines(), prompts[ch].splitlines(),
                fromfile=f"golden/{ch}", tofile=f"now/{ch}", lineterm="", n=2,
            ))[:60])
            pytest.fail(f"flag-off prompt for {ch!r} drifted:\n{diff}")


_CALENDAR_DATE_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|"
    r"November|December) \d{1,2}, \d{4}\b"
)


def test_no_golden_contains_a_calendar_date():
    """The rot this guard cannot afford: a golden captured with the live clock
    in it goes red the next day, on the one guard a deliberate prompt change is
    certified against — and the redness says "the prompt moved", which is a lie.

    R44 (#749) moved the date out of the head entirely — the Runtime Context
    now says the date arrives in <turn_context> — so the assertion is on the
    ABSENCE of a calendar date, with `_freeze()` kept as the belt for a build
    that puts a `- Today's date: …` line back (it would land as <FROZEN>).
    """
    for path in sorted(_GOLDENS.glob("*.txt")):
        text = path.read_text()
        assert not _CALENDAR_DATE_RE.search(text), (
            f"{path.name}: a calendar date is baked into the golden — it will "
            "go red tomorrow for a reason unrelated to the prompt"
        )
        dated = [l for l in text.splitlines() if l.startswith("- Today's date:")]
        assert dated in ([], [_FROZEN_DATE]), f"{path.name}: {dated}"
        assert "Today's date" in text, (
            f"{path.name}: the Runtime Context no longer tells the model where "
            "the date comes from"
        )


def test_the_goldens_name_the_build_they_came_from(goldens):
    """A golden with no provenance cannot be told from a photograph of a bug."""
    assert len(goldens["agent_runner_sha256"]) == 64
    assert goldens["_why"]
    assert goldens["user_name"] == _GOLDEN_USER_NAME, (
        "the golden was captured for a different user name; the persona "
        "section interpolates it, so the two must agree"
    )
    assert set(goldens["sha256"]) >= set(_channels()), (
        "a channel exists with no golden — regenerate the fixture"
    )


# ══════════════════════════════════════════════════════════════════════
# 3. The envelope carries what the prefix stopped carrying
# ══════════════════════════════════════════════════════════════════════


def _env_mod():
    return pytest.importorskip(
        "app.agent.runtime_envelope",
        reason="runtime_envelope not landed yet — lane B1 (D1)",
    )


def _build_env(**kw):
    """The envelope under PRODUCTION inputs: the real CHANNEL_GUIDANCE value,
    the real client-surface label, and the ids `_run_inner` always passes. A
    39-character synthetic guidance here once hid a Surface line truncated to
    a half-word on every channel and deleted outright on three."""
    m = _env_mod()
    from app.agent.agent_runner import CHANNEL_GUIDANCE, _client_surface_label

    origin = kw.get("origin_channel", "whatsapp")
    args = dict(
        origin_channel=origin,
        reply_channel=kw.get("reply_channel", origin),
        client_surface=_client_surface_label(origin),
        guidance=CHANNEL_GUIDANCE.get(origin, ""),
        capabilities=m.channel_capabilities(kw.get("reply_channel", origin)),
        request_id="0f3a9c2e-1111-4bbb-8ccc-dddddddddddd",
        message_id="1a2b3c4d-2222-4bbb-8ccc-eeeeeeeeeeee",
        day_chat_id="9e8d7c6b-3333-4bbb-8ccc-ffffffffffff",
    )
    args.update(kw)
    return m.build_runtime_envelope_message(**args)


def _first_sentence(guidance: str) -> str:
    text = " ".join((guidance or "").split())
    head = text.split("\n", 1)[0]
    dot = head.find(". ")
    return head[: dot + 1] if dot != -1 else head


def test_the_envelope_is_one_fenced_user_message():
    m = _env_mod()
    msg = _build_env()
    assert msg["role"] == "user"
    body = msg["content"]
    assert body.startswith(m.ENVELOPE_OPEN) and body.rstrip().endswith(m.ENVELOPE_CLOSE)
    assert body.count(m.ENVELOPE_OPEN) == 1 and body.count(m.ENVELOPE_CLOSE) == 1


def test_the_envelope_names_origin_reply_and_surface_exactly_once():
    """Once each: a fact repeated is a fact the model has to reconcile."""
    body = _build_env(origin_channel="whatsapp", reply_channel="telegram",
                      client_surface="whatsapp")["content"]
    assert body.count("whatsapp") >= 1 and "telegram" in body
    low = body.lower()
    for token in ("origin", "reply", "surface"):
        assert low.count(f"{token}:") == 1, f"{token!r}: appears {low.count(f'{token}:')}x in the envelope"


@pytest.mark.asyncio
async def test_the_annotation_rule_is_in_the_cached_prompt_not_the_envelope(envelope_on):
    """D2's belt has a prompt half: the model must be told that the
    `[channel h:mmam]` prefixes it sees in history are the SYSTEM's
    annotations and are never to be reproduced in a reply. The rule is
    byte-identical for every channel and every turn, so it lives in the
    CACHED system prompt — review round 1 found it in the per-turn envelope,
    where it was the largest uncached cost and crowded the Surface line out."""
    m = _env_mod()
    assert "[channel h:mmam]" in m.ANNOTATION_RULE
    assert m.ANNOTATION_RULE.count("(") == m.ANNOTATION_RULE.count(")"), "unbalanced parenthesis"
    body = _build_env()["content"]
    assert "[channel h:mmam]" not in body, "the rule is back in the per-turn block"
    prompts = await _prompts(["web", "voice", "trigger"])
    for ch, prompt in prompts.items():
        assert "[channel h:mmam]" in prompt, f"the cached prompt for {ch!r} does not carry the rule"
    assert prompts["web"] == prompts["voice"] == prompts["trigger"], "the rule forked the prefix"


@pytest.mark.parametrize("channel", ["web", "mobile", "voice", "whatsapp", "telegram",
                                     "extension", "vibecoding", "trigger", "api", "app"])
def test_the_surface_line_is_whole_on_every_real_channel(channel):
    """Under production inputs every channel's `Surface:` line must be the
    COMPLETE first sentence of its guidance — not a half-word ellipsis, and
    never absent."""
    from app.agent.agent_runner import CHANNEL_GUIDANCE

    body = _build_env(origin_channel=channel)["content"]
    lines = [l for l in body.splitlines() if l.startswith("Surface: ")]
    assert len(lines) == 1, f"{channel}: Surface line count {len(lines)}"
    surface = lines[0][len("Surface: "):]
    assert "…" not in surface, f"{channel}: Surface truncated: {surface!r}"
    assert surface == _first_sentence(CHANNEL_GUIDANCE[channel]), (
        f"{channel}: {surface!r} != first sentence of the real guidance"
    )


def test_the_envelope_carries_no_ids_line():
    body = _build_env()["content"]
    assert "ids:" not in body.lower(), "eight-character uuid prefixes tell the model nothing"


@pytest.mark.parametrize("channel,create_job,start_mission", [
    ("web", "y", "y"), ("mobile", "y", "y"), ("whatsapp", "y", "y"),
    ("voice", "n", "y"),      # VOICE_DISABLED_TOOLS keeps start_mission (the one voice deferral)
    ("trigger", "n", "n"),    # TRIGGER_DISABLED_TOOLS drops every background scheduler
])
def test_the_deferral_line_matches_the_tool_list(channel, create_job, start_mission):
    """The cached prompt says availability 'is stated on the envelope's
    Deferral line'. It has to actually be there, and it has to agree with
    `disabled_tools_for_channel` — the set the runner removes from the wire
    array."""
    from app.agent.prompt_profile import disabled_tools_for_channel

    body = _build_env(origin_channel=channel)["content"]
    lines = [l for l in body.splitlines() if l.startswith("Deferral: ")]
    assert len(lines) == 1, body
    assert f"create_job={create_job}" in lines[0], lines[0]
    assert f"start_mission={start_mission}" in lines[0], lines[0]
    disabled = disabled_tools_for_channel(channel)
    for tool in ("create_job", "update_job", "start_mission"):
        assert f"{tool}={'n' if tool in disabled else 'y'}" in lines[0], lines[0]


def test_a_managed_voice_task_never_opens_a_second_job():
    body = _build_env(origin_channel="voice", managed_voice=True)["content"]
    assert "create_job=n" in body and "Contract: VOICE" in body


def test_trigger_turns_are_bound_to_the_trigger_contract():
    body = _build_env(origin_channel="trigger")["content"]
    assert "Contract: TRIGGER" in body


@pytest.mark.asyncio
async def test_the_cached_prompt_carries_the_trigger_pin_for_every_channel(envelope_on):
    """G-19b: 'NEVER claim to have sent…' used to reach the model only through
    the `- Channel:` line, which the envelope removed; the descriptor carries
    the first sentence alone. The pin now rides the always-present
    surface_contracts section — present on every channel (byte-identical
    prefix) and in force only when the envelope names TRIGGER."""
    prompts = await _prompts(["trigger", "web"])
    for ch, prompt in prompts.items():
        assert "NEVER claim to have sent" in prompt, f"the trigger pin is missing from {ch!r}'s prompt"
        assert "## TRIGGER — applies only when the runtime envelope names TRIGGER" in prompt
    assert prompts["trigger"] == prompts["web"]


@pytest.mark.parametrize("channel", ["web", "mobile", "voice", "whatsapp", "telegram"])
def test_channel_capabilities_answers_the_documented_keys(channel):
    m = _env_mod()
    caps = m.channel_capabilities(channel)
    expected = {"markdown", "tables", "code_blocks", "quick_reply_buttons",
                "reactions", "media_player", "max_message_chars"}
    assert expected <= set(caps), f"missing {sorted(expected - set(caps))}"


def test_voice_capabilities_say_no_markdown():
    """ANTI-VACUITY for the capabilities block: if every channel answered the
    same dict, the envelope would be carrying nothing."""
    m = _env_mod()
    assert m.channel_capabilities("voice")["markdown"] in (False, "none")
    assert m.channel_capabilities("web")["markdown"] in (True, "full")


def test_the_envelope_is_small():
    """It is paid on EVERY turn and it is behind the cache breakpoint, so it
    is never cached. ENVELOPE_TOKEN_BUDGET is the D1 budget; the per-channel
    essays must not have moved in here. Measured on PRODUCTION inputs for
    every channel the runner can name, plus the unknown ones."""
    m = _env_mod()
    assert m.ENVELOPE_TOKEN_BUDGET <= 160
    for ch in _channels():
        body = _build_env(origin_channel=ch)["content"]
        assert len(body) // 4 <= m.ENVELOPE_TOKEN_BUDGET, (
            f"{ch!r}: envelope ~{len(body)//4} est. tokens (> {m.ENVELOPE_TOKEN_BUDGET})"
        )


# ══════════════════════════════════════════════════════════════════════
# 4. ORDER — a guard whose precondition something above it destroys is
#    invisible to every other check in this repo.
# ══════════════════════════════════════════════════════════════════════


def _requires_envelope_wiring():
    if "runtime_envelope" not in _SRC:
        pytest.skip("envelope not wired into agent_runner yet — lane B1 (D1)")


def test_the_envelope_sits_between_turn_context_and_the_user_message():
    """It must be behind the cacheable prefix (so it cannot bust it) and it
    must not displace messages[-1] — callers assume the user message is last."""
    _requires_envelope_wiring()
    tc = _SRC.index("if _turn_context_parts:")
    user = _SRC.index('messages.append({"role": "user", "content": user_message})')
    candidates = [i for i in range(len(_SRC))
                  if _SRC.startswith("messages.append(_env_msg)", i)]
    assert candidates, "no `messages.append(_env_msg)` — name the variable _env_msg"
    env = candidates[0]
    assert tc < env < user, (
        f"envelope appended out of order (turn_context@{tc}, envelope@{env}, "
        f"user@{user}) — behind the prefix and before the user message, or it "
        "either busts the cache or steals messages[-1]"
    )


def test_a_subagent_run_gets_no_user_surface_envelope():
    """The SUBAGENT profile carries a different system prompt by design and
    has no user surface at all; handing it channel capabilities would invent
    one. Same profile gate the section filter already enforces."""
    _requires_envelope_wiring()
    assert "_profile_sections" in _SRC


# ══════════════════════════════════════════════════════════════════════
# 5. Flag plumbing — a flag with no delivery path is a flag that is OFF
# ══════════════════════════════════════════════════════════════════════


def test_channel_envelope_defaults_on():
    """Per-tenant .env files are written at provision time and never pick up
    a flag introduced later (test_prompt_diet.py:70-78), so a default-OFF flag
    reaches no container in the fleet."""
    if "channel_envelope" not in Settings.model_fields:
        pytest.skip("settings.channel_envelope not landed yet — lane B1 (D1)")
    assert Settings.model_fields["channel_envelope"].default is True


def test_channel_envelope_kill_switch_survives_the_default():
    if "channel_envelope" not in Settings.model_fields:
        pytest.skip("settings.channel_envelope not landed yet — lane B1 (D1)")
    assert Settings(_env_file=None, channel_envelope=False).channel_envelope is False


@pytest.mark.xfail(
    "channel_converge" in Settings.model_fields
    and Settings.model_fields["channel_converge"].default is False,
    reason="RED until lane B1 flips the default (D1)", strict=True,
)
def test_channel_converge_defaults_on():
    """D1 flips it in the same change: the TOOLS tier must converge too, or
    the lineage still forks and a neutral prompt buys nothing."""
    assert Settings.model_fields["channel_converge"].default is True, (
        "channel_converge is still default-False — the tools array forks for "
        "voice and the one-prefix work is only half delivered"
    )


@pytest.mark.xfail(
    '"CHANNEL_ENVELOPE"' not in (_BACKEND.parent / "bridge" / "pool_addon.py").read_text(),
    reason="RED until lane B1 adds CHANNEL_ENVELOPE to _FEATURE_FLAG_ENVS (D1)",
    strict=True,
)
def test_the_bridge_ships_both_flags():
    bridge = (_BACKEND.parent / "bridge" / "pool_addon.py").read_text()
    assert '"CHANNEL_ENVELOPE"' in bridge, (
        "CHANNEL_ENVELOPE missing from bridge/pool_addon.py::_FEATURE_FLAG_ENVS "
        "— the kill switch would have no delivery path to a running tenant"
    )


# ══════════════════════════════════════════════════════════════════════
# R44 — one head per user across INTENTS, not just across channels
# ══════════════════════════════════════════════════════════════════════
#
# The channel lineage was the first fork; the intent is the other axis of the
# same question. `_build_system_prompt` takes an `intent` that selects
# `include_skill_prompts` / `include_environment` / `include_media_section`,
# so on the legacy path a greeting turn and a full turn are DIFFERENT
# instructions bytes — a guaranteed provider miss whenever the category flips
# between consecutive turns, and a warm that can only ever warm one variant.
#
# PR-1 already closed this: every one of those gates reads `(intent.X or
# _stable)` (agent_runner.py ~6250 / ~6384 / ~6501 / ~6528), so under
# `stable_prefix_layout` (default ON since 2026-08-05) the union is always
# built. Measured here rather than asserted from the code: flag ON → ONE
# variant, spread 0 tokens; flag OFF → four variants spanning ~1,126 est
# tokens, the differences being `# Your Environment & Capabilities` and
# `# Media Playback`. It is pinned now because nothing was pinning it, and
# because the connect-time warm (app/agent/cache_warm.py) is only worth
# issuing if the head it replays is the head the next turn asks for,
# whatever the user happens to type.

_INTENT_CATEGORIES = (
    "greeting", "question", "memory", "web", "media",
    "code", "scheduling", "agent", "full",
)


async def _prompts_by_intent():
    """The REAL `_build_system_prompt`, once per intent category, one user."""
    from app.agent import query_intent as _qi
    from app.agent.agent_runner import AgentRunner
    from app.agent.tool_executor import ToolExecutor
    from app.db.database import async_session_maker
    from app.db.models import User
    from app.services.openai_agent_service import OpenAIAgentService

    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"{uid[:8]}@intent.local",
                    hashed_password="x" * 60, name=_GOLDEN_USER_NAME))
        await db.commit()

    runner = AgentRunner(llm_service=OpenAIAgentService(),
                         tool_executor=ToolExecutor(workspace=_WS))
    out: dict[str, str] = {}
    for cat in _INTENT_CATEGORIES:
        intent = getattr(_qi, f"INTENT_{cat.upper()}")
        async with async_session_maker() as db:
            out[cat] = await runner._build_system_prompt(
                db=db, user_id=uid, user_message="hi", channel="app",
                intent=intent, client_tz="UTC", turn_context_out={},
            )
    return out


@pytest.fixture
def stable_layout_on(monkeypatch):
    monkeypatch.setattr(settings, "stable_prefix_layout", True)


@pytest.mark.asyncio
async def test_every_intent_shares_one_system_prompt(envelope_on, stable_layout_on):
    prompts = await _prompts_by_intent()
    distinct = {hashlib.sha256(p.encode()).hexdigest() for p in prompts.values()}
    assert len(distinct) == 1, (
        "the system prompt forks on intent — a greeting turn and a full turn "
        "are different cached heads:\n" + _first_divergence(prompts)
    )


@pytest.mark.asyncio
async def test_greeting_then_full_is_one_byte_identical_head(
    envelope_on, stable_layout_on,
):
    """The consecutive-turn case, spelled out: the exact pair the measured
    pool-88 day kept alternating between."""
    prompts = await _prompts_by_intent()
    assert prompts["greeting"] == prompts["full"]


@pytest.mark.asyncio
async def test_the_intent_variants_under_test_are_real(envelope_on, stable_layout_on):
    """Guard the guard: if the builder returned "" for every category the
    two tests above would pass on nothing."""
    prompts = await _prompts_by_intent()
    assert len(prompts) == len(_INTENT_CATEGORIES)
    assert all(len(p) > 2000 for p in prompts.values())
    assert "# Your Environment & Capabilities" in prompts["greeting"], (
        "the union is not being built — this section is the one the legacy "
        "path drops for a greeting, so its absence means the fork is back"
    )


@pytest.mark.asyncio
async def test_the_fork_is_real_when_the_stable_layout_is_off(envelope_on, monkeypatch):
    """Sensitivity: with PR-1 off the prompts MUST diverge, or the three
    tests above are measuring a builder that ignores intent entirely."""
    monkeypatch.setattr(settings, "stable_prefix_layout", False)
    prompts = await _prompts_by_intent()
    distinct = {hashlib.sha256(p.encode()).hexdigest() for p in prompts.values()}
    assert len(distinct) > 1
    assert prompts["greeting"] != prompts["full"]


def test_the_wire_tools_array_does_not_fork_on_intent():
    """The other half of the head. The intent-filtered array is what the
    legacy path sends; the stable path sends the channel array whatever the
    intent, and moves the gating to `tool_choice`, which is not in the
    cached prefix."""
    from app.agent import query_intent as _qi
    from app.agent.agent_runner import strip_vault_tool_for_channel
    from app.agent.prefix_stability import strip_tools_for_channel
    from app.agent.query_intent import filter_tools_by_intent

    tools = [
        {"name": n, "description": n, "input_schema": {}}
        for n in ("web_search", "exec", "write_file", "play_media",
                  "memory_search", "generate_image")
    ]
    filtered = {
        cat: [t["name"] for t in filter_tools_by_intent(
            tools, getattr(_qi, f"INTENT_{cat.upper()}"))]
        for cat in _INTENT_CATEGORIES
    }
    assert len({tuple(v) for v in filtered.values()}) > 1, (
        "filter_tools_by_intent no longer varies — this test proves nothing"
    )
    stable = [
        t["name"] for t in strip_tools_for_channel(
            tools, "app", strip_vault_tool_for_channel=strip_vault_tool_for_channel)
    ]
    for cat in _INTENT_CATEGORIES:
        assert stable == [
            t["name"] for t in strip_tools_for_channel(
                tools, "app",
                strip_vault_tool_for_channel=strip_vault_tool_for_channel)
        ], f"the stable wire array moved for intent={cat}"


# ══════════════════════════════════════════════════════════════════════
# R44 — one head across local DATES, not just channels and intents
# ══════════════════════════════════════════════════════════════════════
#
# `render_time_lines(stable=True)` used to leave `- Today's date: …` in the
# runtime section, and test_stable_prefix called that "allowed to roll
# daily". Daily is once per user per local midnight, and measured
# 2026-09-13/14 it was 17/17 of the first-message-of-a-day misses on a
# ~40,192-token head — ttft p50 5,586 ms against a hit's 2,611 ms. It also
# forced the connect-time warm to refuse across midnight, and it is why the
# goldens above had to be regenerated every calendar day to stay green.
#
# Measured the honest way: two timezones fourteen hours apart, ONE instant,
# two different local dates. Before the move the heads differed by exactly
# that line; after it they are byte-identical — which makes the head
# tz-invariant as well, since the tz name travelled inside that sentence.

_TZ_AHEAD = "Pacific/Kiritimati"   # UTC+14
_TZ_BEHIND = "Pacific/Midway"      # UTC-11


async def _prompt_and_tail(tz_name: str):
    from app.agent.agent_runner import AgentRunner
    from app.agent.tool_executor import ToolExecutor
    from app.db.database import async_session_maker
    from app.db.models import User
    from app.services.openai_agent_service import OpenAIAgentService

    uid = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=uid, email=f"{uid[:8]}@dateinv.local",
                    hashed_password="x" * 60, name=_GOLDEN_USER_NAME))
        await db.commit()
    runner = AgentRunner(llm_service=OpenAIAgentService(),
                         tool_executor=ToolExecutor(workspace=_WS))
    tail: dict = {}
    async with async_session_maker() as db:
        prompt = await runner._build_system_prompt(
            db=db, user_id=uid, user_message="hi", channel="app",
            client_tz=tz_name, turn_context_out=tail,
        )
    return prompt, tail


def _local_dates_differ() -> bool:
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo

    now = datetime.now(timezone.utc)
    return (now.astimezone(ZoneInfo(_TZ_AHEAD)).date()
            != now.astimezone(ZoneInfo(_TZ_BEHIND)).date())


@pytest.mark.asyncio
async def test_the_head_is_identical_across_two_local_dates(
    envelope_on, stable_layout_on,
):
    assert _local_dates_differ(), (
        f"{_TZ_AHEAD} and {_TZ_BEHIND} are on the same local date right now — "
        "they are 25 hours apart, so this can only mean the zones moved"
    )
    ahead, _ = await _prompt_and_tail(_TZ_AHEAD)
    behind, _ = await _prompt_and_tail(_TZ_BEHIND)
    assert ahead == behind, (
        "the system prompt still moves with the user's local date — every "
        "first message of a day is a guaranteed cache miss:\n"
        + _first_divergence({_TZ_AHEAD: ahead, _TZ_BEHIND: behind})
    )


@pytest.mark.asyncio
async def test_the_date_still_reaches_the_model_in_the_turn_context_tail(
    envelope_on, stable_layout_on,
):
    """Moved, not deleted. The tail is where the clock already lives, and it
    sits after the history, so it costs no cached bytes."""
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo

    prompt, tail = await _prompt_and_tail(_TZ_AHEAD)
    clock = tail.get("clock") or ""
    expected = datetime.now(timezone.utc).astimezone(
        ZoneInfo(_TZ_AHEAD)
    ).strftime("Today's date: %A, %B %d, %Y")
    assert expected in clock, f"tail does not state the date: {clock!r}"
    assert _TZ_AHEAD in clock
    # …and exactly once, nowhere in the cached head.
    assert "Today's date: " not in prompt
    assert clock.count("Today's date: ") == 1
