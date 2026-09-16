"""The requested-format section actually reaches the model — round 46, C9.

`format_intent.requested_format_section` is unit-tested next door
(test_format_intent_contract.py). This file proves the WIRING, which is the
half a pure test cannot see: the section is built inside `_build_system_prompt`
and, under the stable-prefix layout, is written into the per-turn
`<turn_context>` slot rather than into the system prompt — a turn-conditional
block in the prompt body would move the cached prefix for that tenant's every
other turn, and the round's own headline finding is that a cold 40 k prefix
costs ~14 s of user-visible latency.

Two things can silently break it: a `section_parts` key that is not in
`prompt_profile.SECTION_ORDER` is DROPPED at assembly (this is exactly how F1 /
active_tasks went missing for months), and a `turn_context_out` key that run()
does not render is dropped the same way.

agent-mode: `_build_system_prompt` SELECTs `identities`, which is AGENT_ONLY.
"""

from __future__ import annotations

import uuid as _uuid

import pytest


async def _make_user() -> str:
    from app.db import async_session_maker, User

    user_id = str(_uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=user_id, email=f"{user_id[:8]}@test.local",
                    hashed_password="x" * 60, name="Fmt"))
        await db.commit()
    return user_id


def _runner(workdir: str):
    from app.agent.agent_runner import AgentRunner
    from app.agent.tool_executor import ToolExecutor
    from app.services.openai_agent_service import OpenAIAgentService

    return AgentRunner(llm_service=OpenAIAgentService(),
                       tool_executor=ToolExecutor(workspace=workdir))


async def _build(tmp_path, message: str, channel: str = "app"):
    from app.db import async_session_maker

    user_id = await _make_user()
    runner = _runner(str(tmp_path))
    out: dict = {}
    async with async_session_maker() as db:
        prompt = await runner._build_system_prompt(
            db=db, user_id=user_id, user_message=message, channel=channel,
            turn_context_out=out,
        )
    return prompt, out


def _all_text(prompt: str, out: dict) -> str:
    return prompt + "\n" + "\n".join(str(v) for k, v in out.items()
                                     if not k.startswith("__"))


@pytest.mark.asyncio
async def test_a_named_format_reaches_the_model(tmp_path):
    prompt, out = await _build(tmp_path, "export this month's expenses as CSV")
    text = _all_text(prompt, out)
    assert "# Requested format" in text, "the section was built and then dropped"
    assert "generate_data_file" in text
    assert "a CSV file" in text


@pytest.mark.asyncio
async def test_a_turn_that_names_no_format_carries_no_section(tmp_path):
    """It is injected on the turns that name a format and on no others — a
    section on every turn is a section in the cached prefix, which is the one
    thing this must not become."""
    prompt, out = await _build(tmp_path, "what's the weather in Toronto tomorrow")
    assert "# Requested format" not in _all_text(prompt, out)


@pytest.mark.asyncio
async def test_a_video_request_is_refused_in_the_prompt_not_promised(tmp_path):
    """A14: no video producer exists anywhere in the product. The model must be
    told so BEFORE it answers, or it invents a tool call or a promise."""
    prompt, out = await _build(tmp_path, "can you make me a video of this")
    text = _all_text(prompt, out)
    assert "# Requested format" in text
    assert "cannot make one" in text
    section = text[text.index("# Requested format"):]
    section = section.split("\n\n")[0]
    assert "generate_video" not in section


@pytest.mark.asyncio
async def test_the_section_lands_in_the_turn_slot_not_the_cached_prefix(tmp_path):
    from app.config import settings
    if not getattr(settings, "stable_prefix_layout", False):
        pytest.skip("stable prefix layout is off in this configuration")
    prompt, out = await _build(tmp_path, "give me a PDF of that")
    assert "requested_format" in out
    assert "# Requested format" not in prompt, (
        "a turn-conditional block in the system prompt moves the cached prefix "
        "for every other turn of this tenant"
    )


@pytest.mark.asyncio
async def test_the_documents_section_names_every_producer(tmp_path):
    """Both the full guide and the diet must name every tool that exists — a
    producer the standing prompt never mentions is one the model will not
    reach for, which is the same defect round 46 fixed the other way round
    (`csv` in the turn-1 gate with no generator behind it)."""
    prompt, _ = await _build(tmp_path, "hello")
    assert "generate_data_file" in prompt
    assert "generate_audio" in prompt
    # The surface the model is told about must exist on the channel it is on:
    # 'the document pane' is web-only and does not exist on iOS or WhatsApp.
    assert "document pane" not in prompt


@pytest.mark.asyncio
async def test_the_video_refusal_rides_the_turn_section_not_the_standing_prompt(tmp_path):
    """A14's refusal is deliberately NOT in the standing Document Generation
    copy: on the diet path that copy is the compact prompt every turn pays for,
    and a video is asked for rarely. `requested_format_section` carries it on
    exactly the turns that ask — verified above and in
    test_format_intent_contract.py. The FULL copy still states it, because it
    is not paying for its length by the turn."""
    from app.agent.prompt_diet import prompt_diet_enabled

    prompt, out = await _build(tmp_path, "hello")
    body = prompt.lower()
    if prompt_diet_enabled():
        assert "no video generator" not in body, (
            "the diet is the compact prompt — the refusal belongs on the turn "
            "that asks for a video, not on every turn"
        )
    else:
        assert "no video generator" in body
    # Either way, nothing anywhere promises a producer that does not exist.
    assert "generate_video" not in prompt
