"""W1.3 — Browser loop token efficiency (backend/app/api/ws_browser.py).

Pure-function tests for the per-call elided view (old tool results stubbed,
last-2 screenshots only), the per-task trajectory collapse, the prompt-cache
kwargs, and context-overflow detection. No network, no browser.
"""

import copy

from app.api.ws_browser import (
    _browser_cache_kwargs,
    _collapse_finished_task,
    _is_context_length_error,
    _prepare_browser_llm_view,
    _strip_stale_screenshots,
)


def _img_block(tag: str):
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{tag}", "detail": "high"}}


def _tool_turn(i: int, with_image: bool = True):
    """One assistant tool-call + tool-result pair in OpenAI format."""
    assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": f"call_{i}", "type": "function",
             "function": {"name": "click", "arguments": "{}"}}
        ],
    }
    content = [{"type": "text", "text": f"result {i}"}]
    if with_image:
        content.append(_img_block(f"shot{i}"))
    tool = {"role": "tool", "tool_call_id": f"call_{i}", "content": content}
    return [assistant, tool]


def _build_conversation(n_tool_turns: int):
    conv = [{"role": "user", "content": [
        {"type": "text", "text": "find me a flight"},
        _img_block("initial"),
    ]}]
    for i in range(n_tool_turns):
        conv.extend(_tool_turn(i))
    return conv


def _count_images(messages):
    n = 0
    for m in messages:
        if isinstance(m.get("content"), list):
            n += sum(1 for b in m["content"] if isinstance(b, dict) and b.get("type") in ("image_url", "image"))
    return n


def test_view_keeps_only_last_two_images():
    conv = _build_conversation(8)  # 1 user image + 8 tool images
    view = _prepare_browser_llm_view(conv, keep_recent_turns=5, keep_last_images=2)

    assert _count_images(view) == 2
    # The two survivors are the two most recent tool results
    assert any(b.get("type") == "image_url" for b in view[-1]["content"])
    assert any(b.get("type") == "image_url" for b in view[-3]["content"])
    # Stripped messages get a deterministic text stub instead
    stripped_tool = view[2]  # tool turn 0 is inside the elided range
    assert "elided" in str(stripped_tool["content"]) or "omitted" in str(stripped_tool["content"])


def test_view_elides_old_tool_results_keeps_recent_five():
    conv = _build_conversation(8)
    view = _prepare_browser_llm_view(conv, keep_recent_turns=5, keep_last_images=2)

    tool_msgs = [m for m in view if m.get("role") == "tool"]
    assert len(tool_msgs) == 8
    # First 3 elided to string stubs, last 5 keep structured content
    for m in tool_msgs[:3]:
        assert isinstance(m["content"], str) and "[tool result elided" in m["content"]
    for m in tool_msgs[3:]:
        assert isinstance(m["content"], list)
    # Text portion of the kept-recent-but-stripped results survives
    assert any(b.get("type") == "text" and "result 5" in b["text"] for b in tool_msgs[5]["content"])


def test_view_never_mutates_canonical_conversation():
    conv = _build_conversation(8)
    snapshot = copy.deepcopy(conv)
    _prepare_browser_llm_view(conv, keep_recent_turns=5, keep_last_images=2)
    assert conv == snapshot
    assert _count_images(conv) == 9


def test_short_conversation_untouched_except_images():
    conv = _build_conversation(3)  # under the elision threshold
    view = _prepare_browser_llm_view(conv, keep_recent_turns=5, keep_last_images=2)
    tool_msgs = [m for m in view if m.get("role") == "tool"]
    assert all(isinstance(m["content"], list) for m in tool_msgs)
    assert _count_images(view) == 2  # initial user image + turn 0 stripped


def test_strip_handles_string_content_messages():
    msgs = [
        {"role": "user", "content": "plain history string"},
        {"role": "assistant", "content": "reply"},
        {"role": "tool", "tool_call_id": "c1", "content": [{"type": "text", "text": "r"}, _img_block("a")]},
    ]
    out = _strip_stale_screenshots(copy.deepcopy(msgs), keep_last=2)
    assert out[0]["content"] == "plain history string"
    assert _count_images(out) == 1


def test_overflow_retry_window_is_tighter():
    conv = _build_conversation(8)
    view = _prepare_browser_llm_view(conv, keep_recent_turns=2, keep_last_images=2)
    tool_msgs = [m for m in view if m.get("role") == "tool"]
    assert sum(1 for m in tool_msgs if isinstance(m["content"], str)) == 6
    assert _count_images(view) == 2


def test_collapse_finished_task():
    conv = [{"role": "user", "content": "old task"},
            {"role": "assistant", "content": "old summary"}]
    task_start = len(conv)
    conv.extend(_build_conversation(6))  # the finished task's trajectory
    _collapse_finished_task(conv, task_start, "find me a flight", "Cheapest is $412 on United.")
    assert conv == [
        {"role": "user", "content": "old task"},
        {"role": "assistant", "content": "old summary"},
        {"role": "user", "content": "find me a flight"},
        {"role": "assistant", "content": "Cheapest is $412 on United."},
    ]


def test_collapse_without_response_keeps_user_message_only():
    conv = []
    conv.extend(_build_conversation(2))
    _collapse_finished_task(conv, 0, "find me a flight", "")
    assert conv == [{"role": "user", "content": "find me a flight"}]


def test_cache_kwargs_present_for_openai():
    kw = _browser_cache_kwargs("openai", "user-1", "sess-9")
    assert kw == {
        "prompt_cache_key": "browser:user-1:sess-9",
        "prompt_cache_retention": "24h",
        "safety_identifier": "user-1",
    }


def test_cache_kwargs_empty_when_unavailable():
    assert _browser_cache_kwargs("anthropic", "user-1", "sess-9") == {}
    assert _browser_cache_kwargs("openai", None, "sess-9") == {}
    assert _browser_cache_kwargs("openai", "user-1", None) == {}


def test_context_length_error_detection():
    assert _is_context_length_error(Exception(
        "Error code: 400 - {'error': {'message': \"This model's maximum context length is 272000 tokens...\", "
        "'code': 'context_length_exceeded'}}"))
    assert _is_context_length_error(Exception("prompt exceeds the context window"))

    class _CodedErr(Exception):
        code = "context_length_exceeded"

    assert _is_context_length_error(_CodedErr("boom"))
    assert not _is_context_length_error(Exception("rate limit exceeded"))
    assert not _is_context_length_error(Exception("connection reset by peer"))
