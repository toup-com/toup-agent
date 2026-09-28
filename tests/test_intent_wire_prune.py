"""Exact-owner intent wire pilot: routing, capability, cache, and proxy budget.

The 177-definition fixture matches the observed owner's pre-PDF array plus
the two always-included attachment analysis tools (65 core, 37 skills,
60 connector manifest, 10 first-party MCP, 5 unknown). Only the five unknown
definitions and MCP descriptions use stubs. Provider token accounting is
therefore measured separately on paired live turns.
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import pytest

from app.agent.intent_wire_prune import pruned_cache_key, select_first_call_tools
from app.agent.query_intent import classify_query_intent, filter_tools_by_intent
from app.config import Settings, _exact_user_canary_enabled


OWNER = "11111111-1111-4111-8111-111111111111"
OTHER = "22222222-2222-4222-8222-222222222222"
BACKEND = Path(__file__).resolve().parents[1]
RUNNER_SOURCE = (BACKEND / "app/agent/agent_runner.py").read_text()


def _tool(name: str) -> dict:
    return {"name": name, "description": name, "input_schema": {"type": "object"}}


@pytest.mark.parametrize("channel", ["web", "mobile"])
def test_exact_owner_greeting_keeps_the_existing_first_call_names(channel):
    stable = [_tool(name) for name in (
        "navigate_to", "recall_day", "memory_search", "memory_read_file",
        "memory_store", "spawn", "routines__remind", "routines__create",
        "routines__list", "web_search", "exec", "play_media",
    )]
    intent = classify_query_intent("Hi")
    allowed = [t["name"] for t in filter_tools_by_intent(stable, intent)]
    selected = select_first_call_tools(
        stable, allowed, canary_enabled=_exact_user_canary_enabled(OWNER, OWNER),
        channel=channel, intent_category=intent.category, message_text="Hi",
        main_chat=True,
        has_attachment=False, has_job=False,
    )
    assert selected is not None
    assert [t["name"] for t in selected] == allowed
    assert len(selected) == 9


@pytest.mark.parametrize("change", [
    {"canary_enabled": False}, {"channel": "voice"},
    {"channel": "telegram"}, {"intent_category": "full"},
    {"intent_category": "code"}, {"main_chat": False},
    {"has_attachment": True}, {"has_job": True},
])
def test_uncertain_or_noninteractive_turn_uses_legacy_array(change):
    opts = dict(canary_enabled=True, question_canary_enabled=True,
                channel="web", intent_category="question",
                message_text="What is 2+2?",
                main_chat=True, has_attachment=False, has_job=False)
    opts.update(change)
    stable = [_tool("a"), _tool("b")]
    wire_before = json.dumps(stable, ensure_ascii=False)
    assert select_first_call_tools(stable, ["a"], **opts) is None
    assert json.dumps(stable, ensure_ascii=False) == wire_before


def test_missing_duplicate_or_overwide_allow_list_uses_legacy_array():
    stable = [_tool("a"), _tool("b"), _tool("a")]
    opts = dict(canary_enabled=True, question_canary_enabled=True,
                channel="mobile", intent_category="question",
                message_text="What is 2+2?",
                main_chat=True, has_attachment=False, has_job=False)
    assert select_first_call_tools(stable, ["missing"], **opts) is None
    assert select_first_call_tools(stable, ["a", "a"], **opts) is None
    assert select_first_call_tools([_tool(str(i)) for i in range(20)],
                                   [str(i) for i in range(17)], **opts) is None
    selected = select_first_call_tools(stable, ["a"], **opts)
    assert selected == [stable[0]]  # proxy also keeps the first duplicate


@pytest.mark.parametrize('message', [
    'Hi schedule something', 'Hi,schedule something', 'Hi,schedule',
    'Hi:send', 'Hello,send', 'Hey;remind',
])
def test_mixed_greeting_shortcut_stays_on_legacy_schema(message):
    assert select_first_call_tools(
        [_tool('navigate_to'), _tool('routines__create'), _tool('web_search')],
        ['navigate_to', 'routines__create'],
        canary_enabled=True, channel='web', intent_category='greeting',
        message_text=message, main_chat=True, has_attachment=False, has_job=False,
    ) is None


def test_prior_pdf_followup_keeps_locator_and_full_tool_wire():
    from app.agent.attachment_analysis import likely_file_followup

    message = 'Hi, summarize that PDF'
    assert likely_file_followup(message)
    intent = classify_query_intent(message)
    stable = [_tool(name) for name in (
        'navigate_to', 'analyze_attachment', 'read_attachment_analysis',
        'web_search', 'exec',
    )]
    allowed = [t['name'] for t in filter_tools_by_intent(stable, intent)]
    assert {'analyze_attachment', 'read_attachment_analysis'} <= set(allowed)
    assert select_first_call_tools(
        stable, allowed,
        canary_enabled=True, channel='mobile',
        intent_category=intent.category, message_text=message,
        main_chat=True, has_attachment=False, has_job=False,
    ) is None


@pytest.mark.parametrize('message', ['HI', 'hello!', 'Hey...', ' hi, '])
def test_standalone_greeting_spellings_can_prune(message):
    assert select_first_call_tools(
        [_tool('navigate_to'), _tool('web_search')], ['navigate_to'],
        canary_enabled=True, channel='mobile', intent_category='greeting',
        message_text=message, main_chat=True, has_attachment=False, has_job=False,
    ) is not None


def test_opt_in_is_exact_uuid_and_defaults_dark():
    assert Settings.model_fields["intent_wire_prune_canary_user_ids"].default == ""
    assert Settings.model_fields["intent_wire_prune_question_canary_user_ids"].default == ""
    assert _exact_user_canary_enabled(OWNER, f"{OTHER},{OWNER}")
    assert not _exact_user_canary_enabled(OTHER, OWNER)
    assert not _exact_user_canary_enabled(OWNER, OWNER[:8])
    assert not _exact_user_canary_enabled(OWNER, "*")
    assert 'if _pruned_tools is not None:\n                        current_tools = _pruned_tools' in RUNNER_SOURCE


def test_question_requires_second_explicit_dial_and_preserves_web_tools():
    intent = classify_query_intent('What is 2+2?')
    assert intent.category == 'question'
    stable = [_tool(name) for name in (
        'navigate_to', 'recall_day', 'memory_search', 'memory_read_file',
        'memory_store', 'spawn', 'routines__remind', 'routines__create',
        'routines__list', 'web_search', 'web_fetch', 'exec',
    )]
    allowed = [t['name'] for t in filter_tools_by_intent(stable, intent)]
    kwargs = dict(canary_enabled=True, channel='web', intent_category=intent.category,
                  message_text='What is 2+2?', main_chat=True,
                  has_attachment=False, has_job=False)
    assert select_first_call_tools(stable, allowed, **kwargs) is None
    selected = select_first_call_tools(
        stable, allowed, question_canary_enabled=True, **kwargs,
    )
    assert selected is not None
    assert [t['name'] for t in selected] == allowed
    assert {'web_search', 'web_fetch'} <= {t['name'] for t in selected}


def test_pruned_cache_lineage_is_separate_and_restores_full_after_any_tool():
    assert pruned_cache_key(OWNER, "greeting") != f"{OWNER}:all"
    assert pruned_cache_key(OWNER, "greeting") != pruned_cache_key(OWNER, "question")
    assert 'pruned_cache_key(user_id, query_intent.category)' in RUNNER_SOURCE
    assert 'current_tools = _stable_tools' in RUNNER_SOURCE
    assert '_record_cache_warm_head()' in RUNNER_SOURCE
    assert 'intent_wire_prune_restored_head tools=%s sys=%s' in RUNNER_SOURCE
    first_call = RUNNER_SOURCE.split('if _intent_wire_pruned and user_id:', 1)[1]
    assert first_call.index('pruned_cache_key(user_id, query_intent.category)') < first_call.index('_cw_call.set_cache_key')
    after_tool = RUNNER_SOURCE.split('messages.append({"role": "user", "content": tool_results})', 1)[1]
    assert after_tool.index('_restore_full_intent_wire("tool use")') < after_tool.index('if not _stable_prefix')
    tool_choice = RUNNER_SOURCE.split('if (\n                        _stable_prefix', 1)[1]
    assert 'and iteration == 0' in tool_choice.split('):', 1)[0]
    assert 'and not _tool_choice_restriction_rejected' in tool_choice.split('):', 1)[0]


def test_repeat_greeting_has_the_same_cache_head_and_routing_key():
    from app.agent import cache_warm
    from app.agent.prefix_stability import head_hashes

    stable = [_tool('navigate_to'), _tool('recall_day'), _tool('web_search')]
    kwargs = dict(canary_enabled=True, channel='web', intent_category='greeting',
                  message_text='Hi',
                  main_chat=True, has_attachment=False, has_job=False)
    old = cache_warm._HEADS.get(OWNER)
    try:
        observed = []
        for _ in range(2):
            selected = select_first_call_tools(stable, ['navigate_to', 'recall_day'], **kwargs)
            assert selected is not None
            head = head_hashes(selected, 'mandatory identity and safety', [])
            key = pruned_cache_key(OWNER, 'greeting')
            cache_warm.record_head(
                OWNER, llm=object(), system_prompt='mandatory identity and safety',
                tools=selected, model='gpt-5.6-terra', prompt_cache_key=None,
                safety_identifier=OWNER, stable_prefix_active=True,
                channel='web', local_date=date(2026, 9, 24), tz_name='America/Toronto',
            )
            cache_warm.set_cache_key(OWNER, key)
            assert cache_warm._HEADS[OWNER]['tools'] == selected
            assert cache_warm._HEADS[OWNER]['prompt_cache_key'] == key
            observed.append((head, key))
        assert observed[0] == observed[1]
    finally:
        if old is None:
            cache_warm._HEADS.pop(OWNER, None)
        else:
            cache_warm._HEADS[OWNER] = old


def test_provider_rejection_and_model_fallback_restore_full_array():
    rejection = RUNNER_SOURCE.split('and is_tool_choice_rejection(e)', 1)[1]
    assert rejection.index('_restore_full_intent_wire("allowed_tools rejection")') < rejection.index('continue')
    fallback = RUNNER_SOURCE.split('if active_model != fallback:', 1)[1]
    assert fallback.index('_restore_full_intent_wire("model fallback")') < fallback.index('fallback_llm.create_message_stream(')
    assert 'tools=current_tools or None' in fallback


def test_bridge_forwards_the_exact_user_flag():
    bridge = (BACKEND.parent / "bridge/pool_addon.py").read_text()
    flags = bridge.split('_FEATURE_FLAG_ENVS = (', 1)[1].split('\n)', 1)[0]
    assert '"INTENT_WIRE_PRUNE_CANARY_USER_IDS"' in flags
    assert '"INTENT_WIRE_PRUNE_QUESTION_CANARY_USER_IDS"' in flags


@pytest.mark.asyncio
async def test_production_shape_proxy_budget_and_first_call_capability():
    """Measure serialized Responses tools after proxy dedup/cap, not agent count."""
    import tiktoken
    import yaml

    from app.agent.skills.loader import SkillLoader
    from app.agent.tool_definitions import (
        get_agent_tools, get_extended_tools, get_doc_generation_tools,
        get_navigation_tools,
    )
    from app.api.llm_proxy import _cap_tools, _dedup_tool_names
    from app.services.openai_agent_service import _anthropic_tools_to_responses

    core = (get_agent_tools() + get_extended_tools()
            + get_doc_generation_tools() + get_navigation_tools())
    loader = SkillLoader()
    await loader.load_all()
    skills = loader.get_all_tool_definitions()
    manifest_names = []
    for manifest in sorted((BACKEND / 'app/connectors').glob('*/manifest.yaml')):
        data = yaml.safe_load(manifest.read_text()) or {}
        if data.get('id') == 'stub':
            continue
        manifest_names.extend(
            tool.get('name') if isinstance(tool, dict) else tool
            for tool in data.get('tools', [])
        )
    mcp_source = (BACKEND / 'app/mcp_server.py').read_text()
    first_party = re.findall(r'@mcp\.tool\(\)\s*\nasync def (\w+)\(', mcp_source)
    unknown = [f'unattributed__t{i:02d}' for i in range(5)]
    mcp = [
        {"name": name, "description": "x" * 300,
         "input_schema": {"type": "object", "properties": {}}}
        for name in sorted(manifest_names + first_party + unknown)
    ]
    stable = core + skills + mcp
    assert (len(core), len(skills), len(manifest_names), len(first_party), len(stable)) == (
        65, 37, 60, 10, 177,
    )
    intent = classify_query_intent('Hi')
    allowed = sorted({t['name'] for t in filter_tools_by_intent(stable, intent)})
    selected = select_first_call_tools(
        stable, allowed, canary_enabled=True, channel='web',
        intent_category=intent.category, message_text='Hi', main_chat=True,
        has_attachment=False, has_job=False,
    )
    assert selected is not None
    assert {t['name'] for t in selected} == set(allowed)

    full_wire = _anthropic_tools_to_responses(stable)
    deduped, duplicates = _dedup_tool_names(full_wire)
    capped, dropped = _cap_tools(deduped, protected=set(allowed))
    pruned_wire = _anthropic_tools_to_responses(selected)
    assert (len(full_wire), len(deduped), len(capped), len(pruned_wire)) == (177, 176, 128, 11)
    assert 'memory_search' in duplicates
    assert not (set(allowed) & set(dropped))
    assert {t['name'] for t in pruned_wire} == set(allowed)
    assert {'analyze_attachment', 'read_attachment_analysis'} <= {t['name'] for t in pruned_wire}
    encoding = tiktoken.get_encoding('o200k_base')
    tokens = lambda value: len(encoding.encode(json.dumps(value, ensure_ascii=False)))
    full_tokens, pruned_tokens = tokens(capped), tokens(pruned_wire)
    assert full_tokens - pruned_tokens > 10_000
    print(f'POST_PROXY_TOOLS full={len(capped)}:{full_tokens} '
          f'pruned={len(pruned_wire)}:{pruned_tokens} delta={full_tokens - pruned_tokens}')
