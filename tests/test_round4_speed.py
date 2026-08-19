"""Round 4 (2026-08-19) — item 7 SPEED, item 1 favicons, item 8 step attribution.

Measured on the founder's tenant (2026-08-18, `[PERF]` lines): a 17.5 s
"check the latest Gemini model" turn spent 7.2 s in two LLM round-trips whose
only tool calls were `create_job` / `update_job`; a 69 s fine-tuning-cost
turn spent 22.6 s in three of them, plus 14 s in a fetch batch that was
serialised by CPU-bound extraction on the event loop and a 6 s attempt to
launch a browser the image does not ship. Every LLM iteration also awaited a
~0.4 s credit POST before yielding `message_end`, and every search/fetch
awaited a metering POST.

These tests pin the structural changes:
  * `TurnWaterfall` — one structured line per turn, counts bookkeeping-only rounds
  * `StepTracker` — which step an action belongs to (item 8)
  * `extract_web_refs` — the domains/urls a web tool touched (item 1)
  * runner: job tools execute FIRST, before the concurrent web batch; tool
    frames carry call_id / step_index / domains; a reasoning event per round
  * reader: extraction runs OFF the event loop; browser fallback is latched
    off once the binary is proven missing
  * metering / usage reports are scheduled, not awaited
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.agent import agent_runner as AR
from app.agent.agent_runner import AgentRunner, extract_web_refs, _accepts_meta
from app.agent.step_tracker import StepTracker
from app.agent.turn_timing import BOOKKEEPING_TOOLS, TurnWaterfall


_RUNNER_SRC = (Path(__file__).resolve().parent.parent / "app" / "agent" / "agent_runner.py").read_text()
_RUN_INNER_SRC = inspect.getsource(AgentRunner._run_inner)


# ── TurnWaterfall ───────────────────────────────────────────────────────

def test_waterfall_counts_bookkeeping_only_rounds():
    wf = TurnWaterfall()
    wf.note_round(["create_job"])                       # narration only
    wf.note_round(["update_job", "web_search"])         # real work → not counted
    wf.note_round(["update_job"])                       # narration only
    wf.note_round([])                                   # final answer round
    assert wf.tool_rounds == 3
    assert wf.bookkeeping_only_rounds == 2
    assert BOOKKEEPING_TOOLS == frozenset({"create_job", "update_job"})


def test_waterfall_renders_one_json_line_with_stages(caplog):
    caplog.set_level(logging.INFO, logger="app.agent.turn_timing")
    wf = TurnWaterfall()
    wf.start("phase1"); time.sleep(0.01); wf.end("phase1", n_hist=3)
    wf.mark("llm", 120, i=1, ttft_ms=40, out=10, stop="tool_use")
    wf.mark("tool", 30, tool="web_search")
    wf.meta["intent"] = "web"
    data = wf.emit()
    assert data is not None
    line = next(r.getMessage() for r in caplog.records if "[TURN_WATERFALL]" in r.getMessage())
    parsed = json.loads(line.split("[TURN_WATERFALL] ", 1)[1])
    assert parsed["phase1_ms"] >= 10
    assert parsed["llm_ms"] == 120 and parsed["llm_rounds"] == 0  # llm_rounds is bumped by the runner
    assert parsed["tool_ms_sum"] == 30
    assert parsed["intent"] == "web"
    assert [s["name"] for s in parsed["stages"]] == ["phase1", "llm", "tool"]
    # second emit is a no-op
    assert wf.emit() is None


# ── StepTracker (item 8) ────────────────────────────────────────────────

def test_step_tracker_no_job_means_no_fields():
    st = StepTracker()
    assert st.step_index is None
    assert st.event_fields() == {}
    assert st.final_step_index() is None


def test_step_tracker_create_then_advance_attributes_batches_to_the_right_step():
    st = StepTracker()
    changed = st.observe(
        "create_job", {"title": "t", "steps": ["Search", "Read", "Write"]},
        json.dumps({"job_id": "J1", "title": "t", "steps": 3, "job_type": "search"}),
    )
    assert changed and st.job_id == "J1" and st.step_index == 0
    assert st.event_fields() == {
        "job_id": "J1", "step_index": 0, "step_name": "Search",
        "steps_total": 3, "job_type": "search",
    }
    # [update_job(current_step=0), web_fetch, web_fetch] → the fetches are step 1
    changed = st.observe(
        "update_job", {"job_id": "J1", "current_step": 0},
        json.dumps({"ok": True, "status": "running", "completed_steps": 1,
                    "total_steps": 3, "current_step": 1, "job_type": "search"}),
    )
    assert changed and st.step_index == 1 and st.step_name() == "Read"
    # the closing answer round belongs to the LAST (writing) step
    assert st.final_step_index() == 2
    # marking the last step done never runs past the end
    st.observe("update_job", {"job_id": "J1", "current_step": 2},
               json.dumps({"ok": True, "total_steps": 3}))
    assert st.step_index == 2


def test_step_tracker_ignores_errors_and_adopts_foreign_jobs():
    st = StepTracker()
    assert st.observe("create_job", {"title": "t"}, "ERROR: title is required") is False
    assert st.step_index is None
    # an update for a job created elsewhere (regex intake / earlier turn)
    assert st.observe("update_job", {"job_id": "J9", "current_step": 1},
                      json.dumps({"ok": True, "total_steps": 4})) is True
    assert st.job_id == "J9" and st.step_index == 2 and st.steps_total == 4
    # a failed update leaves state alone
    assert st.observe("update_job", {"job_id": "J9", "current_step": 3}, "ERROR: Job J9 not found") is False
    assert st.step_index == 2


def test_step_tracker_terminal_status_lands_on_the_last_step():
    st = StepTracker()
    st.observe("create_job", {"steps": ["a", "b"]}, json.dumps({"job_id": "J", "steps": 2}))
    st.observe("update_job", {"job_id": "J", "status": "completed"}, json.dumps({"ok": True, "total_steps": 2}))
    assert st.step_index == 1


# ── extract_web_refs (item 1) ───────────────────────────────────────────

def test_extract_web_refs_input_url_first_then_result_urls_deduped():
    result = (
        "1. Gemini 3.7 (published: 2026-08-01)\n   https://blog.google/products/gemini/x\n"
        "2. Docs\n   https://ai.google.dev/gemini-api/docs/models\n"
        "3. Again\n   https://www.blog.google/other?utm=1.\n"
    )
    domains, urls = extract_web_refs("web_search", {"query": "gemini"}, result)
    assert domains == ["blog.google", "ai.google.dev"]
    assert urls[0] == "https://blog.google/products/gemini/x"
    assert urls[2] == "https://www.blog.google/other?utm=1"   # trailing '.' trimmed

    domains, urls = extract_web_refs("web_fetch", {"url": "https://huggingface.co/docs/peft"}, "# PEFT\nbody")
    assert domains == ["huggingface.co"] and urls == ["https://huggingface.co/docs/peft"]


def test_extract_web_refs_error_results_ground_nothing_beyond_input():
    domains, urls = extract_web_refs("web_fetch", {"url": "https://x.example/a"}, "ERROR: Could not read https://x.example/a")
    assert domains == ["x.example"] and urls == ["https://x.example/a"]
    assert extract_web_refs("web_search", {}, "ERROR: gateway down") == ([], [])
    # cap
    many = "\n".join(f"https://site{i}.example/p" for i in range(30))
    d, u = extract_web_refs("web_search", {}, many)
    assert len(d) == 10 and len(u) == 10


# ── _accepts_meta ───────────────────────────────────────────────────────

def test_accepts_meta_detects_meta_kwarg_or_var_kwargs_only():
    async def old(name, summary, tool_input=None): ...
    async def new(name, summary, tool_input=None, meta=None): ...
    async def kw(name, **kwargs): ...
    assert _accepts_meta(old) is False
    assert _accepts_meta(new) is True
    assert _accepts_meta(kw) is True
    assert _accepts_meta(None) is False


# ── runner: source pins for the tool-round invariants ───────────────────

def _code_only(src: str) -> str:
    return "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))


def test_job_bookkeeping_runs_with_the_web_batch_and_the_loop_stamps_after():
    """The prompt asks the model to put create_job/update_job in the SAME
    response as the searches. The runner runs the job chain CONCURRENTLY
    with the web batch (create_job is 0.6–1 s of DB+notify; paying it in
    front of the searches was ~1.5 s/turn) and the ORDERED loop — which
    runs after both — is what stamps step attribution on the frames."""
    src = _code_only(_RUN_INNER_SRC)
    i_bk = src.index("async def _run_bookkeeping() -> None:")
    i_gather = src.index("await asyncio.gather(_run_bookkeeping(), _batch_coro)")
    i_loop = src.index("for tc in pending_tool_calls:\n                if cancel_check and cancel_check():")
    assert i_bk < i_gather < i_loop
    # the pre-executed job tools are consumed by the ordered loop like the
    # parallel results are (no double execution)
    assert '_parallel_results[tc["id"]] = {' in src
    # a lone web call still overlaps with bookkeeping
    assert "_run_batch = len(_parallel_tcs) > 1 or (bool(_parallel_tcs) and bool(_bk_tcs))" in src


def test_tool_end_frames_carry_call_id_step_and_domains():
    src = _RUN_INNER_SRC
    assert 'await on_tool_end(tc["name"], summary, tc.get("input"), meta=_meta)' in src
    for key in ('"call_id": tc["id"]', '"elapsed_ms": int(_elapsed_ms)', "**_step_fields", '_meta["domains"] = _domains'):
        assert key in src, key
    # persisted records get the same attribution so history re-renders match live
    assert '"call_id": tc["id"],\n                    "started_at_ms": _t_tool_started_ms' in src
    # tool_start carries the (provisional) step + call id
    assert 'meta={"call_id": event.tool_id, **_steps.event_fields()}' in src


def test_a_reasoning_event_is_emitted_per_llm_round_and_final_round_lands_on_last_step():
    src = _RUN_INNER_SRC
    assert '"kind": "reasoning"' in src
    assert "_steps.final_step_index() if _is_final_round else _steps.step_index" in src
    assert '"kind": "step_change"' in src


def test_thinking_status_fires_at_run_start_not_after_phase1():
    """7d: the client's live indicator must not wait behind phase 1."""
    src = _code_only(_RUN_INNER_SRC)
    i_status = src.index('await on_status("thinking")')
    i_phase1 = src.index("Phase 1: Load from DB") if "Phase 1: Load from DB" in src else src.index('_wf.start("phase1")')
    assert i_status < i_phase1


def test_waterfall_is_emitted_at_turn_end():
    assert "_wf.emit()" in _RUN_INNER_SRC
    assert "[TURN_WATERFALL]" in (Path(__file__).resolve().parent.parent / "app" / "agent" / "turn_timing.py").read_text()


def test_bookkeeping_only_rounds_are_counted_for_prod_compliance():
    assert '_wf.note_round([tc["name"] for tc in pending_tool_calls])' in _RUN_INNER_SRC


# ── the prompt contract (S1) ────────────────────────────────────────────

def test_job_tool_contract_forbids_bookkeeping_only_responses():
    from app.agent import tool_definitions as td
    src = inspect.getsource(td)
    for phrase in (
        "call create_job IN THE SAME RESPONSE as the",
        "never in a response by itself",
        "do NOT call update_job to mark the",
        "a response containing only update_job is a wasted model",
    ):
        assert phrase in src, phrase


def test_prompt_rules_no_longer_say_create_job_first_alone():
    from app.agent.prompt_diet import _DIET_JOB_RULE
    rule = _DIET_JOB_RULE(False)
    assert "IN THE SAME" in rule and "never mark completed" in rule
    assert "`create_job` first, then" not in rule
    assert "call `create_job` FIRST, then `update_job` as you complete each step" not in _RUNNER_SRC


# ── reader: extraction off the loop, browser latch ──────────────────────

def test_reader_extraction_runs_in_a_worker_thread(monkeypatch):
    """The parse+extract for one page took 1–6 s and BLOCKED THE EVENT
    LOOP, serialising the 'parallel' fetch batch. Prove the sync half runs
    in a worker thread: it must not run on the loop's thread."""
    import threading
    from app.agent.smart_fetch import reader as R

    seen = {}
    real = R._parse_and_extract

    def spy(html, url, max_chars):
        seen["thread"] = threading.current_thread().name
        return real(html, url, max_chars)

    monkeypatch.setattr(R, "_parse_and_extract", spy)
    monkeypatch.setattr(R.settings, "fetch_cache_enabled", False)

    class _Resp:
        text = "<html><head><title>T</title></head><body><p>" + ("article body " * 40) + "</p></body></html>"
        headers = {"content-type": "text/html"}
        is_redirect = False
        url = "http://example.com/a"
        def raise_for_status(self): pass

    class _Client:
        def __init__(self, *a, **k): pass
        async def get(self, url, headers=None): return _Resp()

    monkeypatch.setattr(R.httpx, "AsyncClient", _Client)
    monkeypatch.setattr(R, "_assert_public_url", lambda url: None)

    async def go():
        main = threading.current_thread().name
        out = await R.toup_read_page("http://example.com/a", 10000)
        return main, out

    main, out = asyncio.run(go())
    assert "article body" in out
    assert seen["thread"] != main, "extraction ran on the event-loop thread"


def test_reader_uses_lxml_when_available():
    from app.agent.smart_fetch import reader as R
    try:
        import lxml  # noqa: F401
        assert R._bs_parser() == "lxml"
    except ImportError:
        assert R._bs_parser() == "html.parser"


def test_reader_shares_one_client_per_loop(monkeypatch):
    from app.agent.smart_fetch import reader as R
    made = {"n": 0}

    class _Client:
        def __init__(self, *a, **k):
            made["n"] += 1
            self.is_closed = False

    monkeypatch.setattr(R.httpx, "AsyncClient", _Client)
    monkeypatch.setattr(R, "_SHARED_CLIENT", None)
    monkeypatch.setattr(R, "_SHARED_CLIENT_LOOP", None)

    async def go():
        a = R._client(); b = R._client()
        return a is b

    assert asyncio.run(go()) is True
    assert made["n"] == 1
    # a new loop → a new client (the old loop is closed)
    asyncio.run(go())
    assert made["n"] == 2


def test_browser_launch_failure_for_missing_binary_is_latched(monkeypatch):
    from app.agent.skills.builtins.app_builder import browser_api as B
    monkeypatch.setattr(B, "_UNAVAILABLE_REASON", None)
    monkeypatch.setattr(B, "_browser", None)

    class _Chromium:
        async def launch(self, **kw):
            raise Exception(
                "BrowserType.launch: Executable doesn't exist at /root/.cache/ms-playwright/"
                "chromium_headless_shell-1228/chrome-headless-shell\n"
                "Please run the following command to download new browsers:\n    patchright install"
            )

    class _PW:
        chromium = _Chromium()
        async def stop(self): pass

    class _Starter:
        async def start(self): return _PW()

    import types, sys
    fake_mod = types.ModuleType("patchright.async_api")
    fake_mod.async_playwright = lambda: _Starter()
    pkg = types.ModuleType("patchright"); pkg.async_api = fake_mod
    monkeypatch.setitem(sys.modules, "patchright", pkg)
    monkeypatch.setitem(sys.modules, "patchright.async_api", fake_mod)

    async def go():
        with pytest.raises(RuntimeError):
            await B._ensure_browser()
        # second call short-circuits BEFORE touching the driver
        with pytest.raises(RuntimeError) as ei:
            await B._ensure_browser()
        return str(ei.value)

    msg = asyncio.run(go())
    assert B.browser_unavailable_reason() is not None
    assert "not available" in msg


def test_web_fetch_skips_the_browser_when_latched(monkeypatch):
    from app.agent import tool_executor as TE
    from app.agent.skills.builtins.app_builder import browser_api as B
    monkeypatch.setattr(B, "_UNAVAILABLE_REASON", "browser executable missing")
    monkeypatch.setattr(TE.settings, "browser_fetch_enabled", True, raising=False)
    monkeypatch.setattr(TE.settings, "browser_fetch_latch_missing", True, raising=False)
    monkeypatch.setattr(TE.settings, "web_token_budget_enabled", False, raising=False)

    from app.agent.smart_fetch import reader as R
    monkeypatch.setattr(R, "_assert_public_url", lambda url: None)
    monkeypatch.setattr(R, "page_cache_get", lambda url, mc: None)

    async def _empty(url, max_chars):  # "JS-rendered" signal
        return ""
    monkeypatch.setattr(R, "toup_read_page", _empty)

    called = {"browser": False}
    async def _read_page(url):
        called["browser"] = True
        return "rendered"
    monkeypatch.setattr(B, "read_page", _read_page)

    ex = TE.ToolExecutor.__new__(TE.ToolExecutor)
    ex.set_user_id("u1")
    ex.set_channel("web")
    out = asyncio.run(ex._tool_web_fetch({"url": "https://example.com/js"}))
    assert out.startswith("ERROR: Could not read")
    assert called["browser"] is False, "the dead browser fallback must not be attempted"


# ── metering / usage off the critical path ──────────────────────────────

def test_web_metering_is_scheduled_not_awaited(monkeypatch):
    from app.agent import tool_executor as TE
    monkeypatch.setattr(TE.settings, "web_tool_metering_enabled", True, raising=False)
    ran = {"n": 0}

    async def slow_bg(self, *a, **k):
        await asyncio.sleep(0.2)
        ran["n"] += 1
    monkeypatch.setattr(TE.ToolExecutor, "_meter_web_tool_bg", slow_bg)

    ex = TE.ToolExecutor.__new__(TE.ToolExecutor)
    ex.set_user_id("u1")
    ex.set_channel("web")

    async def go():
        t0 = time.perf_counter()
        await ex._meter_web_tool("web_search", tier="gateway", engine="brave", started=time.monotonic(), query="q")
        dt = time.perf_counter() - t0
        assert dt < 0.1, f"metering was awaited inline ({dt:.2f}s)"
        await asyncio.sleep(0.3)
        assert ran["n"] == 1
    asyncio.run(go())


def test_llm_usage_report_is_scheduled_not_awaited(monkeypatch):
    from app.services import credit_reporter as CR
    calls = []

    async def fake_report(**kw):
        await asyncio.sleep(0.05)
        calls.append(kw)
    monkeypatch.setattr(CR, "report_llm_usage", fake_report)

    async def go():
        t0 = time.perf_counter()
        CR.report_llm_usage_bg(user_id="u", model="m", provider="openai", input_tokens=1, output_tokens=1)
        assert time.perf_counter() - t0 < 0.02
        assert calls == []
        await asyncio.sleep(0.1)
        assert len(calls) == 1 and calls[0]["model"] == "m"
    asyncio.run(go())


def test_stream_paths_use_the_background_reporter():
    from app.services import openai_agent_service as O, anthropic_service as A
    for mod in (O, A):
        src = inspect.getsource(mod)
        assert "await report_llm_usage(" not in src, mod.__name__
        assert "report_llm_usage_bg(" in src, mod.__name__


def test_gateway_search_reuses_one_client_per_loop(monkeypatch):
    from app.agent import tool_executor as TE
    made = {"n": 0}

    class _Client:
        def __init__(self, *a, **k):
            made["n"] += 1
            self.is_closed = False
    monkeypatch.setattr(TE.httpx, "AsyncClient", _Client)
    TE._reset_gateway_client()

    async def go():
        return TE._gateway_client() is TE._gateway_client()
    assert asyncio.run(go()) is True
    assert made["n"] == 1
    TE._reset_gateway_client()


def test_query_embedding_is_bounded_on_the_turn_path():
    from app.services import memory_service as M
    src = inspect.getsource(M.MemoryService.hybrid_search)
    assert "asyncio.wait_for(" in src and "memory_embed_timeout_s" in src
    from app.config import settings
    assert 0 < settings.memory_embed_timeout_s <= 5


# ── ws_chat frames ─────────────────────────────────────────────────────

def test_ws_frames_carry_the_new_fields():
    src = (Path(__file__).resolve().parent.parent / "app" / "api" / "ws_chat.py").read_text()
    assert 'async def on_tool_start(tool_name: str, meta: Optional[dict] = None):' in src
    assert 'meta: Optional[dict] = None):' in src  # on_tool_end
    assert '{"type": "step_event", **(payload or {})}' in src
    assert "on_step_event=on_step_event," in src
    for k in ('"call_id"', '"step_index"', '"domains"', '"urls"', '"job_type"'):
        assert k in src, k


# ── end-to-end: a real turn through run() with a fake LLM + fake tools ──
#
# Same harness shape as test_dropped_span_promotion_gate.py: SUBAGENT profile
# (no Conversation row → no agent-only tables), history + prompt stubbed, the
# real ToolExecutor with `execute` replaced. What is real: _run_inner's tool
# round — ordering, attribution, frames, waterfall.

class _ScriptedLLM:
    """Round 1: create_job + two web_searches in ONE response. Round 2:
    update_job(current_step=0) + a web_fetch. Round 3: the answer."""

    def __init__(self):
        self.calls = 0

    async def create_message_stream(self, **kwargs):
        from app.services.openai_agent_service import StreamEvent as E
        self.calls += 1
        if self.calls == 1:
            batch = [
                ("c1", "create_job", {"title": "Compare X and Y", "steps": ["Search", "Read", "Write"]}),
                ("s1", "web_search", {"query": "x"}),
                ("s2", "web_search", {"query": "y"}),
            ]
        elif self.calls == 2:
            batch = [
                ("u1", "update_job", {"job_id": "JOB1", "current_step": 0}),
                ("f1", "web_fetch", {"url": "https://docs.example.com/x"}),
            ]
        else:
            yield E(type="text", text="done **bold**")
            yield E(type="message_end", stop_reason="end_turn", usage={"input_tokens": 10, "output_tokens": 2})
            return
        for cid, name, _ in batch:
            yield E(type="tool_use_start", tool_name=name, tool_id=cid)
        for cid, name, inp in batch:
            yield E(type="tool_use_end", tool_name=name, tool_id=cid, tool_input=inp)
        yield E(type="message_end", stop_reason="tool_use", usage={"input_tokens": 10, "output_tokens": 5})


async def _make_user_r4() -> str:
    import uuid as _uuid
    from app.db import async_session_maker, User
    user_id = str(_uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(id=user_id, email=f"{user_id[:8]}@test.local", hashed_password="x" * 60, name="R4"))
        await db.commit()
    return user_id


@pytest.mark.asyncio
async def test_end_to_end_job_tools_first_and_frames_attributed(monkeypatch, tmp_path):
    import uuid as _uuid
    import app.agent.agent_runner as ar
    from app.agent.prompt_profile import PromptProfile
    from app.agent.tool_executor import ToolExecutor

    async def _no_history(self, db, session_id, max_messages: int = 50, client_tz=None):
        return []

    async def _fixed_prompt(self, *a, **kw):
        return "You are Toup."

    monkeypatch.setattr(ar.AgentRunner, "_load_history", _no_history)
    monkeypatch.setattr(ar.AgentRunner, "_build_system_prompt", _fixed_prompt)
    monkeypatch.setattr(ar, "_spawn_background", lambda coro: coro.close())
    monkeypatch.setattr(ar, "_spawn_bg", lambda coro, **k: coro.close())
    monkeypatch.setattr(ar.settings, "citation_gate_enabled", False, raising=False)

    order: list[str] = []
    in_flight = {"n": 0, "max": 0}

    async def fake_execute(name, inp):
        order.append(name)
        if name == "create_job":
            await asyncio.sleep(0.05)
            return json.dumps({"job_id": "JOB1", "title": inp["title"], "steps": 3, "job_type": "compare"})
        if name == "update_job":
            return json.dumps({"ok": True, "status": "running", "completed_steps": 1,
                               "total_steps": 3, "current_step": 1, "job_type": "compare"})
        if name == "web_search":
            in_flight["n"] += 1
            in_flight["max"] = max(in_flight["max"], in_flight["n"])
            await asyncio.sleep(0.05)
            in_flight["n"] -= 1
            return f"1. Result for {inp['query']}\n   https://{inp['query']}.example.org/page\n"
        if name == "web_fetch":
            return "# Docs\nbody text"
        return "?"

    tools = ToolExecutor(workspace=str(tmp_path))
    monkeypatch.setattr(tools, "execute", fake_execute)

    starts, ends, steps, statuses = [], [], [], []

    async def on_tool_start(name, meta=None):
        starts.append((name, dict(meta or {})))

    async def on_tool_end(name, summary, tool_input=None, meta=None):
        ends.append((name, dict(meta or {})))

    async def on_step_event(payload):
        steps.append(dict(payload))

    async def on_status(stage):
        statuses.append(stage)

    runner = ar.AgentRunner(llm_service=_ScriptedLLM(), tool_executor=tools)  # type: ignore[arg-type]
    user_id = await _make_user_r4()
    resp = await runner.run(
        user_message="compare X and Y", user_id=user_id,
        session_id=f"subagent:{_uuid.uuid4()}", channel="subagent",
        prompt_profile=PromptProfile.SUBAGENT,
        save_user_message=False, save_assistant_message=False,
        disable_post_processing=True, model_override="gpt-5.5-mini",
        on_tool_start=on_tool_start, on_tool_end=on_tool_end,
        on_step_event=on_step_event, on_status=on_status,
    )
    assert resp.text == "done **bold**"

    # 1. bookkeeping ran BEFORE the web batch in each round
    assert order[:3] == ["create_job", "web_search", "web_search"], order
    assert order[3:5] == ["update_job", "web_fetch"], order
    # 2. the two searches overlapped (still concurrent after the reorder)
    assert in_flight["max"] == 2
    # 3. 'thinking' was sent immediately at run start (before any LLM call)
    assert statuses[0] == "thinking"
    # 4. tool_end frames: call_id + authoritative step attribution + domains
    by_id = {m["call_id"]: (n, m) for n, m in ends}
    assert by_id["s1"][1]["step_index"] == 0 and by_id["s1"][1]["job_id"] == "JOB1"
    assert by_id["s1"][1]["step_name"] == "Search" and by_id["s1"][1]["steps_total"] == 3
    assert by_id["s1"][1]["domains"] == ["x.example.org"]
    assert by_id["f1"][1]["step_index"] == 1 and by_id["f1"][1]["step_name"] == "Read"
    assert by_id["f1"][1]["domains"] == ["docs.example.com"]
    assert by_id["c1"][1]["job_id"] == "JOB1" and "elapsed_ms" in by_id["c1"][1]
    # 5. tool_start frames carry the call id (pairing) — provisional step
    assert {m.get("call_id") for _, m in starts} == {"c1", "s1", "s2", "u1", "f1"}
    # 6. step events: a step_change per bookkeeping call, a reasoning event
    #    per LLM round, and the closing round lands on the LAST step
    kinds = [s["kind"] for s in steps]
    assert kinds.count("step_change") == 2
    reasoning = [s for s in steps if s["kind"] == "reasoning"]
    assert [r["iteration"] for r in reasoning] == [1, 2, 3]
    assert reasoning[1]["step_index"] == 0          # thought during step 0 (before update_job ran)
    assert reasoning[2]["final"] is True and reasoning[2]["step_index"] == 2
    assert reasoning[2]["step_name"] == "Write"


@pytest.mark.asyncio
async def test_end_to_end_waterfall_line_and_legacy_callbacks_still_work(monkeypatch, tmp_path, caplog):
    """Old two/three-positional callbacks (Telegram, voice, ws_router) must
    keep working byte-for-byte; and the turn must log ONE waterfall line
    with bookkeeping_only_rounds=0 for a compliant model."""
    import uuid as _uuid
    import app.agent.agent_runner as ar
    from app.agent.prompt_profile import PromptProfile
    from app.agent.tool_executor import ToolExecutor

    async def _no_history(self, db, session_id, max_messages: int = 50, client_tz=None):
        return []

    async def _fixed_prompt(self, *a, **kw):
        return "You are Toup."

    monkeypatch.setattr(ar.AgentRunner, "_load_history", _no_history)
    monkeypatch.setattr(ar.AgentRunner, "_build_system_prompt", _fixed_prompt)
    monkeypatch.setattr(ar, "_spawn_background", lambda coro: coro.close())
    monkeypatch.setattr(ar, "_spawn_bg", lambda coro, **k: coro.close())
    monkeypatch.setattr(ar.settings, "citation_gate_enabled", False, raising=False)

    async def fake_execute(name, inp):
        if name == "create_job":
            return json.dumps({"job_id": "JOB1", "steps": 3, "job_type": "compare"})
        if name == "update_job":
            return json.dumps({"ok": True, "total_steps": 3})
        return "ok https://a.example/1"

    tools = ToolExecutor(workspace=str(tmp_path))
    monkeypatch.setattr(tools, "execute", fake_execute)

    legacy_starts, legacy_ends = [], []

    async def on_tool_start(tool_name):            # legacy shape: no meta
        legacy_starts.append(tool_name)

    async def on_tool_end(tool_name, summary, tool_input=None):   # legacy shape
        legacy_ends.append(tool_name)

    caplog.set_level(logging.INFO, logger="app.agent.turn_timing")
    runner = ar.AgentRunner(llm_service=_ScriptedLLM(), tool_executor=tools)  # type: ignore[arg-type]
    user_id = await _make_user_r4()
    resp = await runner.run(
        user_message="compare X and Y", user_id=user_id,
        session_id=f"subagent:{_uuid.uuid4()}", channel="subagent",
        prompt_profile=PromptProfile.SUBAGENT,
        save_user_message=False, save_assistant_message=False,
        disable_post_processing=True, model_override="gpt-5.5-mini",
        on_tool_start=on_tool_start, on_tool_end=on_tool_end,
    )
    assert resp.text == "done **bold**"
    assert legacy_starts == ["create_job", "web_search", "web_search", "update_job", "web_fetch"]
    assert legacy_ends == legacy_starts
    lines = [r.getMessage() for r in caplog.records if "[TURN_WATERFALL]" in r.getMessage()]
    assert len(lines) == 1
    wf = json.loads(lines[0].split("[TURN_WATERFALL] ", 1)[1])
    assert wf["llm_rounds"] == 3 and wf["tool_rounds"] == 2
    assert wf["bookkeeping_only_rounds"] == 0
    assert wf["job"] == "JOB1" and wf["steps_total"] == 3
    names = [s["name"] for s in wf["stages"]]
    assert names.count("llm") == 3 and "tool_batch" in names and "phase1" in names


def test_reader_single_parse_path_uses_the_lxml_tree_for_check_meta_and_extract(monkeypatch):
    """Prod has trafilatura; CI/local do not. Exercise the tree branch with a
    fake trafilatura whose extract() must receive the PARSED TREE (not the
    HTML string) — that is the whole point: one parse, not two."""
    import sys, types
    LH = pytest.importorskip("lxml.html")   # python-docx pulls lxml in CI; guard anyway
    from app.agent.smart_fetch import reader as R

    seen = {}
    fake = types.ModuleType("trafilatura")
    fake.load_html = lambda html: LH.fromstring(html)
    def _extract(obj, **kw):
        seen["arg_type"] = type(obj).__name__
        return "the article body"
    fake.extract = _extract
    monkeypatch.setitem(sys.modules, "trafilatura", fake)

    html = ("<html><head><title>Plain</title><meta property='og:title' content='OG Title'>"
            "<meta name='author' content='Ada'><meta property='article:published_time' content='2026-08-01'>"
            "<script>var x = '" + "y" * 500 + "';</script></head><body><p>" + "visible text " * 30 + "</p></body></html>")
    status, out = R._parse_and_extract(html, "https://x.example/a", 10000)
    assert status == "ok"
    assert seen["arg_type"] == "HtmlElement", "trafilatura must be handed the parsed tree"
    assert out.startswith("# OG Title\nAuthor: Ada\nDate: 2026-08-01\n\nthe article body")

    # a JS shell: big <script>, no visible body text → "js" (scripts do not count)
    shell = "<html><head><script>" + "z" * 5000 + "</script></head><body><div id='root'></div></body></html>"
    status, _ = R._parse_and_extract(shell, "https://x.example/b", 10000)
    assert status == "js"


def test_reader_falls_back_to_beautifulsoup_without_trafilatura(monkeypatch):
    import sys
    from app.agent.smart_fetch import reader as R
    monkeypatch.setitem(sys.modules, "trafilatura", None)   # import raises ImportError
    html = "<html><head><title>T</title></head><body><article><p>" + "article body " * 40 + "</p></article></body></html>"
    status, out = R._parse_and_extract(html, "https://x.example/c", 10000)
    assert status == "ok" and out.startswith("# T\n\n") and "article body" in out


def test_browser_precheck_reads_the_drivers_browsers_json(monkeypatch, tmp_path):
    """The one-time doomed launch cost ~5 s of node-driver startup per
    container start (every blue-green rollout). The pre-check answers from
    the filesystem: the driver's browsers.json names the revision it wants;
    if that dir is absent under the browsers path, the latch arms without
    spawning anything. Fleet finding 2026-08-19: /opt/toup/playwright holds
    revision 1223, patchright 1.61.2 wants 1228."""
    import json, sys, types
    from app.agent.skills.builtins.app_builder import browser_api as B

    pkg_dir = tmp_path / "patchright"
    (pkg_dir / "driver" / "package").mkdir(parents=True)
    (pkg_dir / "driver" / "package" / "browsers.json").write_text(json.dumps({
        "browsers": [{"name": "chromium", "revision": "1228"},
                     {"name": "chromium-headless-shell", "revision": "1228"}]}))
    fake_pkg = types.ModuleType("patchright"); fake_pkg.__file__ = str(pkg_dir / "__init__.py")
    monkeypatch.setitem(sys.modules, "patchright", fake_pkg)
    browsers = tmp_path / "ms-playwright"; browsers.mkdir()
    (browsers / "chromium_headless_shell-1223").mkdir()          # the stale one
    monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", str(browsers))
    monkeypatch.setattr(B, "_UNAVAILABLE_REASON", None)

    reason = B._installed_browser_missing()
    assert reason and "1228" in reason and "chromium_headless_shell-1223" in reason
    assert B.browser_unavailable_reason() is not None          # latch armed, no launch

    # install the wanted revision → the pre-check clears (self-heals on the next process)
    (browsers / "chromium_headless_shell-1228").mkdir()
    monkeypatch.setattr(B, "_UNAVAILABLE_REASON", None)
    assert B._installed_browser_missing() is None
    assert B.browser_unavailable_reason() is None
