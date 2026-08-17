"""Regression tests for the VOICE_REALTIME_V2 relay upgrade.

Pin two things:
  * Flag OFF ⇒ the session config / URL are byte-identical to v1
    (server_vad 0.8/700ms, whisper-1, gen-1 model, no truncation key) —
    the flag is the rollback, so v1 drift here is a release blocker.
  * Flag ON ⇒ the exact GA shapes verified against OpenAI docs 2026-07-17
    (semantic_vad nesting, truncation.retention_ratio, cached-vs-uncached
    audio/text pricing math from response.done usage blocks).
"""

from __future__ import annotations

import pytest

from app.config import settings
import app.api.ws_realtime as rt


@pytest.fixture
def v2_on(monkeypatch):
    monkeypatch.setattr(settings, "voice_realtime_v2", True)


@pytest.fixture
def v2_off(monkeypatch):
    monkeypatch.setattr(settings, "voice_realtime_v2", False)


# ── Session config shape ──────────────────────────────────────────────

def test_v1_session_config_unchanged(v2_off):
    cfg = rt.build_session_config("INSTR", [{"type": "function", "name": "t"}], "coral")
    assert cfg["type"] == "session.update"
    session = cfg["session"]
    assert session["instructions"] == "INSTR"
    td = session["audio"]["input"]["turn_detection"]
    assert td == {
        "type": "server_vad",
        "threshold": 0.8,
        "prefix_padding_ms": 300,
        "silence_duration_ms": 700,
    }
    # No `language` key at all when the hint is unknown — an ABSENT key is what
    # leaves Whisper on auto-detect. Sending language=null would be equivalent
    # to the API, but the absence is the contract the rest of the suite pins.
    assert session["audio"]["input"]["transcription"] == {"model": "whisper-1"}
    assert session["audio"]["output"]["voice"] == "coral"
    assert "truncation" not in session
    assert rt.realtime_url().endswith("model=gpt-realtime")
    assert rt.realtime_model() == "gpt-realtime"


def test_v2_session_config(v2_on):
    cfg = rt.build_session_config("INSTR", [], "marin")
    session = cfg["session"]
    td = session["audio"]["input"]["turn_detection"]
    assert td["type"] == "semantic_vad"
    assert td["eagerness"] == "auto"
    assert td["create_response"] is True
    assert td["interrupt_response"] is True
    assert session["audio"]["input"]["transcription"] == {
        "model": settings.voice_realtime_transcription_model,
        "prompt": rt.transcription_prompt(None),
    }
    assert session["audio"]["input"]["noise_reduction"] == {"type": "far_field"}
    assert session["truncation"] == {"type": "retention_ratio", "retention_ratio": 0.8}
    assert session["audio"]["output"]["voice"] == "marin"
    assert rt.realtime_url().endswith(f"model={settings.voice_realtime_model}")


# ── Usage → cost math ─────────────────────────────────────────────────
# The usage block below is OpenAI's documented response.done example
# (developers.openai.com/api/docs/guides/realtime-costs).

DOCS_USAGE = {
    "total_tokens": 253,
    "input_tokens": 132,
    "output_tokens": 121,
    "input_token_details": {
        "text_tokens": 119,
        "audio_tokens": 13,
        "image_tokens": 0,
        "cached_tokens": 64,
        "cached_tokens_details": {"text_tokens": 64, "audio_tokens": 0, "image_tokens": 0},
    },
    "output_token_details": {"text_tokens": 30, "audio_tokens": 91},
}


def test_usage_to_cost_cents_docs_example():
    # gpt-realtime-2.1: uncached text in = 119-64 = 55 @ $4/M; cached 64 @ $0.40/M;
    # audio in 13 @ $32/M; audio out 91 @ $64/M; text out 30 @ $24/M.
    expected_usd = (13 * 32.0 + 64 * 0.40 + 55 * 4.0 + 91 * 64.0 + 30 * 24.0) / 1_000_000
    cents = rt._usage_to_cost_cents("gpt-realtime-2.1", DOCS_USAGE)
    assert cents == pytest.approx(expected_usd * 100, rel=1e-9)


def test_cached_audio_billed_at_cached_rate():
    usage = {
        "input_tokens": 12_000,
        "output_tokens": 0,
        "input_token_details": {
            "text_tokens": 0,
            "audio_tokens": 12_000,
            "cached_tokens": 10_000,
            "cached_tokens_details": {"text_tokens": 0, "audio_tokens": 10_000},
        },
        "output_token_details": {"text_tokens": 0, "audio_tokens": 0},
    }
    # 2,000 uncached @ $32/M + 10,000 cached @ $0.40/M
    expected_usd = (2_000 * 32.0 + 10_000 * 0.40) / 1_000_000
    assert rt._usage_to_cost_cents("gpt-realtime-2.1", usage) == pytest.approx(
        expected_usd * 100, rel=1e-9
    )


def test_mini_is_cheaper_and_unknown_model_falls_back_to_flagship():
    flagship = rt._usage_to_cost_cents("gpt-realtime-2.1", DOCS_USAGE)
    mini = rt._usage_to_cost_cents("gpt-realtime-2.1-mini", DOCS_USAGE)
    unknown = rt._usage_to_cost_cents("gpt-realtime-99-experimental", DOCS_USAGE)
    assert mini < flagship
    assert unknown == pytest.approx(flagship, rel=1e-9)


def test_empty_usage_costs_nothing():
    assert rt._usage_to_cost_cents("gpt-realtime-2.1", {}) == 0.0


# ── Instructions budget trimming ─────────────────────────────────────

def _section(header: str, n: int, line: str) -> str:
    return "\n".join([header] + [f"- {line} {i}" for i in range(n)])


def test_cap_chars_noop_under_budget():
    text = _section("# H", 3, "memory")
    assert rt._cap_chars(text, 10_000, keep="head") == text
    assert rt._cap_chars(text, 0, keep="head") == text  # 0 = unbudgeted


def test_cap_chars_head_keeps_highest_priority_entries():
    text = _section("# Agent Brain", 200, "fact")
    capped = rt._cap_chars(text, 300, keep="head")
    assert len(capped) <= 300 + len("- [context trimmed to budget]") + 1
    lines = capped.split("\n")
    assert lines[0] == "# Agent Brain"
    assert lines[1] == "- fact 0"          # earliest (highest-priority) entries survive
    assert lines[-1] == "- [context trimmed to budget]"
    assert "- fact 199" not in capped


def test_cap_chars_tail_keeps_newest_messages():
    text = _section("# Today", 200, "msg")
    capped = rt._cap_chars(text, 300, keep="tail")
    lines = capped.split("\n")
    assert lines[0] == "# Today"
    assert lines[1] == "- [context trimmed to budget]"
    assert lines[-1] == "- msg 199"        # newest messages survive
    assert "- msg 0" not in capped


# ── Full-parity `think` (V2): route to the user's OWN full agent over HTTP ──
# On platform-api the in-process agent_runner is absent, so v1's `think` fell
# back to a TOOL-LESS model call. V2 routes `think` to the user's agent
# /api/chat (the same AgentRunner text chat runs) with save=False, giving voice
# the identical toolset/skills/connectors without double-writing the day-chat.


async def test_think_v2_runs_full_agent_via_http(v2_on, monkeypatch):
    monkeypatch.setattr(rt, "_agent_runner", None)

    class _Decision:
        model = "gpt-5.5"

    monkeypatch.setattr(
        "app.services.model_router.classify_request", lambda task: _Decision()
    )

    async def _fake_vps_info(user_id):
        return ("https://u.agents.toup.ai", "agent-key")

    monkeypatch.setattr(rt, "_get_vps_info", _fake_vps_info)

    calls = []

    async def _fake_vps_api(agent_url, agent_api_key, method, path,
                            params=None, json_body=None, timeout=15.0):
        calls.append(dict(method=method, path=path, json_body=json_body, timeout=timeout))
        return {"text": "The newest is X.", "model": "gpt-5.5", "tool_calls": 2}

    monkeypatch.setattr(rt, "_vps_api", _fake_vps_api)

    text, model = await rt._think("user-1", "what's the newest model?", "sess-9")

    assert text == "The newest is X."
    assert model == "gpt-5.5"
    assert len(calls) == 1
    c = calls[0]
    assert c["method"] == "POST"
    # MUST be the full-agent endpoint, NOT /api/chat (that route is tool-less +
    # always-persists — see api_v1.internal_agent_turn / ws_realtime._think).
    assert c["path"] == "/api/v1/internal/agent-turn"
    assert c["json_body"]["message"] == "what's the newest model?"
    assert c["json_body"]["session_id"] == "sess-9"
    assert c["json_body"]["save"] is False              # voice handler owns persistence
    assert c["timeout"] == settings.voice_realtime_think_timeout_s


async def test_think_v1_does_not_use_agent_http(v2_off, monkeypatch):
    monkeypatch.setattr(rt, "_agent_runner", None)

    class _Decision:
        model = "claude-x"     # Option B → Anthropic branch (stubbed, no network)

    monkeypatch.setattr(
        "app.services.model_router.classify_request", lambda task: _Decision()
    )

    async def _fake_vps_info(user_id):
        return ("https://u.agents.toup.ai", "agent-key")

    monkeypatch.setattr(rt, "_get_vps_info", _fake_vps_info)

    paths = []

    async def _fake_vps_api(agent_url, agent_api_key, method, path,
                            params=None, json_body=None, timeout=15.0):
        paths.append((method, path))
        return []              # empty session history for the context read

    monkeypatch.setattr(rt, "_vps_api", _fake_vps_api)

    class _FakeEvent:
        type = "text"
        text = "fallback answer"

    class _FakeAnthropic:
        async def create_message_stream(self, **kwargs):
            yield _FakeEvent()

    monkeypatch.setattr(
        "app.services.anthropic_service.AnthropicService", lambda: _FakeAnthropic()
    )

    text, _model = await rt._think("user-1", "hello?", "sess-9")

    assert text == "fallback answer"
    # parity path skipped when flag off
    assert ("POST", "/api/v1/internal/agent-turn") not in paths


def test_v2_per_user_allowlist(monkeypatch):
    """Per-account rollout: an allowlisted user gets V2 while the global flag is
    still off; others don't. Global flag on ⇒ everyone gets V2."""
    monkeypatch.setattr(settings, "voice_realtime_v2", False)
    monkeypatch.setattr(settings, "voice_realtime_v2_user_ids", " u-123 , u-456 ")
    assert rt._resolve_v2_for_user("u-123") is True
    assert rt._resolve_v2_for_user("u-456") is True
    assert rt._resolve_v2_for_user("u-999") is False
    assert rt._resolve_v2_for_user(None) is False
    monkeypatch.setattr(settings, "voice_realtime_v2", True)
    assert rt._resolve_v2_for_user("u-999") is True


# ── Transcription language hint ───────────────────────────────────────
# Regression cover for the Persian-voice bug: Whisper re-detects language per
# utterance, so a short Persian utterance carrying an English proper noun came
# back as "Premi is Sabroki." and the inner agent turn answered the garbage
# rather than the request. The pin is session-level and evidence-gated —
# guessing wrong is worse than not guessing, so "unknown" must stay unset.

FA_UTTERANCE = "خب باستم آهنگ یه آهنگ دیگه ازش رو پلی کن."
FA_MIXED = "الان آهنگ Praise The Lord از A$AP Rocky داره پخش می‌شه."
EN_UTTERANCE = "Play me the new track by Kendrick Lamar tomorrow morning."


def _voice_msg(content, role="user", channel="voice"):
    return {"role": role, "channel": channel, "content": content}


@pytest.fixture
def hint_on(monkeypatch):
    monkeypatch.setenv(rt._LANG_HINT_ENV, "true")
    rt._lang_cache.clear()
    yield
    rt._lang_cache.clear()


def _stub_vps(monkeypatch, days, messages_by_date):
    """Stand in for the user's agent: a day-chat list plus per-day messages."""
    async def fake_vps_info(user_id):
        return ("https://agent.test", "key")

    async def fake_vps_api(url, key, method, path, params=None, json_body=None, timeout=15.0):
        if path == "/api/day-chats":
            return days
        for date, msgs in messages_by_date.items():
            if path == f"/api/day-chats/{date}/messages":
                return msgs
        return None

    monkeypatch.setattr(rt, "_get_vps_info", fake_vps_info)
    monkeypatch.setattr(rt, "_vps_api", fake_vps_api)


# ── The pin itself lands in the session payload ───────────────────────

def test_language_absent_when_unknown_v1_and_v2(v2_off):
    """No hint ⇒ byte-identical to today. This is the no-regression pin."""
    v1 = rt.build_session_config("I", [], "coral")["session"]
    assert "language" not in v1["audio"]["input"]["transcription"]
    assert rt.build_session_config("I", [], "coral", None)["session"] == v1


def test_language_rides_the_prompt_on_v2_never_the_pin(v2_on):
    """The V2 transcriber gets the session language as a PROMPT, never as a
    `language` pin. Measured 2026-08-16 (scripts/eval_voice_transcription.py):
    a fa pin on gpt-4o-transcribe TRANSLATED an English utterance into Farsi,
    while the Farsi bias prompt recovered code-switched product names 9/9."""
    tr = rt.build_session_config("I", [], "marin", "fa")["session"]["audio"]["input"]["transcription"]
    assert "language" not in tr
    assert tr["model"] == settings.voice_realtime_transcription_model
    assert "فارسی" in tr["prompt"]          # the fa prompt, in Farsi
    assert "Grok" in tr["prompt"]           # bias terms present


def test_v2_prompt_is_english_when_language_unknown(v2_on):
    tr = rt.build_session_config("I", [], "marin", None)["session"]["audio"]["input"]["transcription"]
    assert "language" not in tr
    assert "فارسی" not in tr["prompt"]
    assert "Grok" in tr["prompt"]


def test_language_pins_on_v1_whisper_too(v2_off):
    tr = rt.build_session_config("I", [], "coral", "fa")["session"]["audio"]["input"]["transcription"]
    assert tr == {"model": "whisper-1", "language": "fa"}


# ── Per-message script detection ──────────────────────────────────────

def test_script_detection_ignores_a_quoted_foreign_title():
    """A Persian song title inside an English sentence is not a Persian speaker."""
    assert rt.detect_script_language(
        "Play the song آ please", min_share=0.30, min_chars=6,
    ) is None
    assert rt.detect_script_language(EN_UTTERANCE, min_share=0.30, min_chars=6) is None


def test_script_detection_accepts_persian_with_inline_latin():
    """Mixed Persian + Latin proper nouns is the normal shape of these turns."""
    assert rt.detect_script_language(FA_UTTERANCE, min_share=0.30, min_chars=6) == "fa"
    assert rt.detect_script_language(FA_MIXED, min_share=0.30, min_chars=6) == "fa"


# ── Resolution from history ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_resolves_fa_from_minority_of_voice_messages(monkeypatch, hint_on):
    """PRESENCE, not majority. The bug corrupts most Persian turns into Latin
    gibberish, so only a minority survive in script — 2 of 16 here. A majority
    rule would never fire for the user this exists to fix."""
    msgs = [_voice_msg(FA_UTTERANCE), _voice_msg(FA_MIXED, role="assistant")]
    msgs += [_voice_msg("Play me shot mirror") for _ in range(14)]
    _stub_vps(
        monkeypatch,
        [{"local_date": "2026-07-31", "message_count": 44, "channels_active": ["mobile", "voice"]}],
        {"2026-07-31": msgs},
    )
    assert await rt.resolve_voice_language("u-fa") == "fa"


@pytest.mark.asyncio
async def test_english_only_speaker_gets_no_hint(monkeypatch, hint_on):
    """The no-regression case: forcing fa onto English audio produces
    hallucinated Persian, so an English speaker must resolve to None."""
    _stub_vps(
        monkeypatch,
        [{"local_date": "2026-07-31", "message_count": 20, "channels_active": ["voice"]}],
        {"2026-07-31": [_voice_msg(EN_UTTERANCE) for _ in range(20)]},
    )
    assert await rt.resolve_voice_language("u-en") is None


@pytest.mark.asyncio
async def test_single_foreign_message_is_below_the_floor(monkeypatch, hint_on):
    """One Persian turn in a long English window is a song title, not a speaker."""
    msgs = [_voice_msg(FA_UTTERANCE)] + [_voice_msg(EN_UTTERANCE) for _ in range(19)]
    _stub_vps(
        monkeypatch,
        [{"local_date": "2026-07-31", "message_count": 20, "channels_active": ["voice"]}],
        {"2026-07-31": msgs},
    )
    assert await rt.resolve_voice_language("u-one") is None


@pytest.mark.asyncio
async def test_typed_channels_are_not_evidence(monkeypatch, hint_on):
    """Scoped to voice: this user types English and speaks Persian, so counting
    mobile/web turns reports the wrong language."""
    _stub_vps(
        monkeypatch,
        [{"local_date": "2026-07-31", "message_count": 4, "channels_active": ["voice"]}],
        {"2026-07-31": [
            _voice_msg(FA_UTTERANCE, channel="mobile"),
            _voice_msg(FA_UTTERANCE, channel="mobile"),
            _voice_msg(EN_UTTERANCE),
            _voice_msg(EN_UTTERANCE),
        ]},
    )
    assert await rt.resolve_voice_language("u-typed") is None


@pytest.mark.asyncio
async def test_days_without_voice_are_never_fetched(monkeypatch, hint_on):
    """channels_active rides the list payload, so silent days cost no request."""
    fetched = []

    async def fake_vps_info(user_id):
        return ("https://agent.test", "key")

    async def fake_vps_api(url, key, method, path, params=None, json_body=None, timeout=15.0):
        if path == "/api/day-chats":
            return [
                {"local_date": "2026-07-31", "message_count": 4, "channels_active": ["voice"]},
                {"local_date": "2026-07-30", "message_count": 0, "channels_active": []},
                {"local_date": "2026-07-29", "message_count": 6, "channels_active": ["mobile"]},
            ]
        fetched.append(path)
        return [_voice_msg(FA_UTTERANCE), _voice_msg(FA_MIXED, role="assistant")]

    monkeypatch.setattr(rt, "_get_vps_info", fake_vps_info)
    monkeypatch.setattr(rt, "_vps_api", fake_vps_api)
    assert await rt.resolve_voice_language("u-days") == "fa"
    assert fetched == ["/api/day-chats/2026-07-31/messages"]


@pytest.mark.asyncio
async def test_no_agent_means_no_hint(monkeypatch, hint_on):
    async def no_vps(user_id):
        return None
    monkeypatch.setattr(rt, "_get_vps_info", no_vps)
    assert await rt.resolve_voice_language("u-novps") is None


# ── Kill switch + caching ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_kill_switch_disables_without_a_deploy(monkeypatch):
    rt._lang_cache.clear()
    monkeypatch.setenv(rt._LANG_HINT_ENV, "false")

    async def boom(user_id):
        raise AssertionError("must not touch the agent when disabled")

    monkeypatch.setattr(rt, "_get_vps_info", boom)
    assert await rt.resolve_voice_language("u-off") is None
    assert rt._lang_hint_enabled() is False
    monkeypatch.setenv(rt._LANG_HINT_ENV, "true")
    assert rt._lang_hint_enabled() is True


@pytest.mark.asyncio
async def test_resolution_is_per_session_not_per_utterance(monkeypatch, hint_on):
    """Resolved once and cached: a voice call must not re-read history per turn."""
    calls = []
    _stub_vps(
        monkeypatch,
        [{"local_date": "2026-07-31", "message_count": 4, "channels_active": ["voice"]}],
        {"2026-07-31": [_voice_msg(FA_UTTERANCE), _voice_msg(FA_MIXED, role="assistant")]},
    )
    real = rt._detect_voice_language

    async def counting(user_id):
        calls.append(user_id)
        return await real(user_id)

    monkeypatch.setattr(rt, "_detect_voice_language", counting)
    assert await rt.resolve_voice_language("u-cache") == "fa"
    assert await rt.resolve_voice_language("u-cache") == "fa"
    assert await rt.resolve_voice_language("u-cache") == "fa"
    assert len(calls) == 1
    # The cache is also what the first session.update reads, with no I/O.
    assert rt._cached_voice_language("u-cache") == "fa"
    assert rt._cached_voice_language("u-never-seen") is None
