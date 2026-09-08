"""Matched deterministic mobile-chat benchmark for the actual AgentRunner.

The provider and tools are controlled fixtures; the runner, scheduling, intent
classifier, operation cache, and database lifecycle are real. No network or
paid model call is made. Point ``TOUP_BENCH_SOURCE`` at a checkout so the same
harness can compare a baseline commit with a candidate patch.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
import time
import uuid
from pathlib import Path


os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
SOURCE = Path(os.environ.get("TOUP_BENCH_SOURCE", Path(__file__).parents[2])).resolve()
sys.path.insert(0, str(SOURCE / "backend"))

import app.agent.agent_runner as ar  # noqa: E402
from app.agent.prompt_profile import PromptProfile  # noqa: E402
from app.agent.query_intent import classify_query_intent, filter_tools_by_intent  # noqa: E402
from app.agent.tool_executor import ToolExecutor  # noqa: E402
from app.db import User, async_session_maker, drop_db, init_db  # noqa: E402
from app.db.database import engine  # noqa: E402
from app.services.openai_agent_service import StreamEvent  # noqa: E402


CONTROLLED_SOURCE_URL = "https://example.test/source"
CONTROLLED_FETCH_TEXT = "Controlled supporting detail"


def _content_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(
            _content_text(item.get("text") if isinstance(item, dict) else item)
            for item in value
        )
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _tool_results(messages) -> dict[str, str]:
    """Extract only executed tool-result blocks, never proposed arguments."""
    results: dict[str, str] = {}
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "tool_result":
                continue
            call_id = str(item.get("tool_use_id") or "")
            if call_id:
                results[call_id] = _content_text(item.get("content"))
    return results


def _is_error_result(content: str) -> bool:
    return not content.strip() or "ERROR:" in content.upper()


class ScriptedLLM:
    def __init__(self, rounds):
        self.rounds = list(rounds)
        self.index = 0
        self.messages = []

    async def create_message_stream(self, **kwargs):
        self.messages.append(kwargs["messages"])
        if self.index < len(self.rounds):
            calls = self.rounds[self.index]
            self.index += 1
            for call_id, name, payload in calls:
                yield StreamEvent(
                    type="tool_use_start", tool_name=name, tool_id=call_id,
                )
                yield StreamEvent(
                    type="tool_use_end", tool_name=name, tool_id=call_id,
                    tool_input=payload,
                )
            yield StreamEvent(
                type="message_end", stop_reason="tool_use",
                usage={"input_tokens": 20, "output_tokens": 10},
            )
            return
        tool_results = _tool_results(kwargs["messages"])
        complete = bool(tool_results) and not any(
            _is_error_result(content) for content in tool_results.values()
        )
        yield StreamEvent(type="text", text=(
            "Complete with all requested evidence and artifact. "
            f"Source: {CONTROLLED_SOURCE_URL}"
            if complete else
            "Could not complete because required controlled evidence failed."
        ))
        yield StreamEvent(
            type="message_end", stop_reason="end_turn",
            usage={"input_tokens": 20, "output_tokens": 10},
        )


async def no_history(self, db, session_id, max_messages=50, client_tz=None):
    return []


async def fixed_prompt(self, *args, **kwargs):
    return "You are Toup. Use every requested result and preserve citations."


def install_runner_fixtures() -> None:
    ar.AgentRunner._load_history = no_history
    ar.AgentRunner._build_system_prompt = fixed_prompt
    ar._spawn_background = lambda coro: coro.close()
    ar._spawn_bg = lambda coro, **kwargs: coro.close()
    ar.settings.citation_gate_enabled = False
    ar.settings.agent_parallel_tool_cap = 2


async def seed_user() -> str:
    user_id = str(uuid.uuid4())
    async with async_session_maker() as db:
        db.add(User(
            id=user_id,
            email=f"agent-runner-bench-{user_id[:8]}@example.test",
            hashed_password="x" * 60,
            name="Benchmark User",
        ))
        await db.commit()
    return user_id


def scenarios():
    return {
        "simple_search_english": {
            "message": "Search the latest AI news with sources",
            "rounds": [[
                ("search-en", "web_search", {"query": "latest AI news"}),
            ]],
            "logical_calls": 1,
            "expected_queries": ["latest AI news"],
            "requires_source": True,
        },
        "simple_search_persian": {
            "message": "آخرین اخبار هوش مصنوعی را جستجو کن و با منبع بده",
            "rounds": [[
                ("search-fa", "web_search", {"query": "آخرین اخبار هوش مصنوعی"}),
            ]],
            "logical_calls": 1,
            "expected_queries": ["آخرین اخبار هوش مصنوعی"],
            "requires_source": True,
        },
        "three_independent_reads": {
            "message": "Search three independent current sources",
            "rounds": [[
                ("search-a", "web_search", {"query": "alpha"}),
                ("search-b", "web_search", {"query": "beta"}),
                ("search-c", "web_search", {"query": "gamma"}),
            ]],
            "logical_calls": 3,
            "expected_queries": ["alpha", "beta", "gamma"],
            "requires_source": True,
        },
        "duplicate_safe_read_with_distinct_read": {
            "message": "Search the current evidence without duplicate work",
            "rounds": [[
                ("search-a", "web_search", {"query": "alpha", "count": 5}),
                ("search-a-copy", "web_search", {"count": 5, "query": "alpha"}),
                ("search-b", "web_search", {"query": "beta", "count": 5}),
            ]],
            "logical_calls": 3,
            "expected_queries": ["alpha", "beta"],
            "requires_source": True,
        },
        "duplicate_connector_read": {
            "message": "List today's routines once and use the result twice",
            "rounds": [[
                ("list-a", "routines__list", {"day": "today"}),
                ("list-a-copy", "routines__list", {"day": "today"}),
            ]],
            "logical_calls": 2,
        },
        "live_status_progression": {
            "message": "Keep checking the build until it completes",
            "rounds": [
                [("status-running", "app_builder__get_status", {"job_id": "build-live"})],
                [("status-completed", "app_builder__get_status", {"job_id": "build-live"})],
            ],
            "logical_calls": 2,
            "expected_result_text": {
                "status-running": "Status: running",
                "status-completed": "Status: completed",
            },
        },
        "multi_step_research_with_artifact": {
            "message": "Research the topic across sources and create a report",
            "rounds": [
                [
                    ("search-a", "web_search", {"query": "alpha"}),
                    ("search-b", "web_search", {"query": "beta"}),
                    ("search-c", "web_search", {"query": "gamma"}),
                ],
                [
                    ("read-a", "web_fetch", {"url": "https://example.test/a"}),
                    ("read-a-copy", "web_fetch", {"url": "https://example.test/a"}),
                    ("read-b", "web_fetch", {"url": "https://example.test/b"}),
                ],
                [
                    ("artifact", "write_file", {
                        "path": "report.md", "content": "all matched evidence",
                    }),
                ],
            ],
            "logical_calls": 7,
            "expected_queries": ["alpha", "beta", "gamma"],
            "requires_source": True,
        },
    }


async def run_once(user_id: str, workspace: str, spec: dict, delay: float) -> dict:
    llm = ScriptedLLM(spec["rounds"])
    tools = ToolExecutor(workspace=workspace)
    executed: list[dict] = []
    expected_calls = [call for round_calls in spec["rounds"] for call in round_calls]
    workspace_path = Path(workspace).resolve()
    for _, name, payload in expected_calls:
        if name == "write_file":
            target = (workspace_path / str(payload.get("path") or "")).resolve()
            target.relative_to(workspace_path)
            target.unlink(missing_ok=True)

    failure_tool = spec.get("failure_tool")

    async def execute(name, payload):
        executed.append({"name": name, "input": dict(payload)})
        await asyncio.sleep(delay)
        if name == failure_tool:
            return f"ERROR: controlled {name} failure"
        if name == "write_file":
            relative = str(payload.get("path") or "")
            target = (workspace_path / relative).resolve()
            target.relative_to(workspace_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            content = str(payload.get("content") or "")
            target.write_text(content, encoding="utf-8")
            return json.dumps({
                "fixture": "local_artifact",
                "path": relative,
                "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            }, sort_keys=True)
        if name == "web_search":
            return json.dumps({
                "query": payload.get("query"),
                "results": [{
                    "title": "Matched evidence",
                    "url": CONTROLLED_SOURCE_URL,
                    "snippet": "Controlled current evidence",
                }],
            }, ensure_ascii=False, sort_keys=True)
        if name == "web_fetch":
            return json.dumps({
                "url": payload.get("url"),
                "text": CONTROLLED_FETCH_TEXT,
            }, sort_keys=True)
        if name == "app_builder__get_status":
            observations = sum(
                item["name"] == name for item in executed
            )
            return "Status: running" if observations == 1 else "Status: completed"
        return f"evidence:{name}:{json.dumps(payload, sort_keys=True)}"

    tools.execute = execute
    runner = ar.AgentRunner(llm_service=llm, tool_executor=tools)
    started = time.perf_counter()
    response = await runner.run(
        user_message=spec["message"],
        user_id=user_id,
        session_id=str(uuid.uuid4()),
        channel="mobile",
        prompt_profile=PromptProfile.FULL,
        model_override="gpt-5.5-mini",
        save_user_message=False,
        save_assistant_message=False,
        disable_post_processing=True,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    result_map = _tool_results(llm.messages[-1])

    def result_matches(call_id: str, name: str, payload: dict, content: str) -> bool:
        if _is_error_result(content):
            return False
        expected_text = spec.get("expected_result_text", {}).get(call_id)
        if expected_text:
            return expected_text in content
        if name == "web_search":
            return (
                json.dumps(payload.get("query"), ensure_ascii=False) in content
                and CONTROLLED_SOURCE_URL in content
                and "Controlled current evidence" in content
            )
        if name == "web_fetch":
            return (
                json.dumps(payload.get("url"), ensure_ascii=False) in content
                and CONTROLLED_FETCH_TEXT in content
            )
        if name == "write_file":
            expected_hash = hashlib.sha256(
                str(payload.get("content") or "").encode("utf-8")
            ).hexdigest()
            return (
                '"fixture": "local_artifact"' in content
                and json.dumps(str(payload.get("path") or "")) in content
                and expected_hash in content
            )
        return (
            f"evidence:{name}:" in content
            and json.dumps(payload, sort_keys=True) in content
        )

    result_checks = {
        call_id: result_matches(call_id, name, payload, result_map.get(call_id, ""))
        for call_id, name, payload in expected_calls
    }
    expected_ids = {call_id for call_id, _, _ in expected_calls}
    artifact_checks = []
    for _, name, payload in expected_calls:
        if name != "write_file":
            continue
        target = (workspace_path / str(payload.get("path") or "")).resolve()
        artifact_checks.append(
            target.is_file()
            and target.read_text(encoding="utf-8") == str(payload.get("content") or "")
        )
    executed_queries = {
        item["input"].get("query")
        for item in executed
        if item["name"] == "web_search"
    }
    sample = {
        "elapsed_ms": elapsed_ms,
        "underlying_calls": len(executed),
        "logical_results": len(result_map),
        "complete": response.text.startswith("Complete with all requested evidence"),
        "all_logical_results_present": set(result_map) == expected_ids,
        "required_evidence_passed": all(result_checks.values()),
        "failed_result_ids": sorted(
            call_id for call_id, passed in result_checks.items() if not passed
        ),
        "artifact_verified": all(artifact_checks) if artifact_checks else True,
        "queries_preserved": executed_queries == set(spec.get("expected_queries", [])),
    }
    sample["quality_passed"] = all((
        sample["complete"],
        sample["all_logical_results_present"],
        sample["required_evidence_passed"],
        sample["artifact_verified"],
        sample["queries_preserved"],
        sample["logical_results"] == spec["logical_calls"],
    ))
    return sample


def nearest_rank(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


async def main(iterations: int, delay_ms: float) -> None:
    install_runner_fixtures()
    await init_db()
    user_id = await seed_user()
    output = {}
    try:
        with tempfile.TemporaryDirectory(prefix="toup-agent-runner-bench-") as workspace:
            scenario_specs = scenarios()
            oracle = {}
            for label, failed_tool in {
                "failed_fetch_rejected": "web_fetch",
                "failed_artifact_rejected": "write_file",
            }.items():
                failed_spec = {
                    **scenario_specs["multi_step_research_with_artifact"],
                    "failure_tool": failed_tool,
                }
                failed_sample = await run_once(user_id, workspace, failed_spec, 0)
                oracle[label] = {
                    "rejected": not failed_sample["quality_passed"],
                    "failed_result_ids": failed_sample["failed_result_ids"],
                    "artifact_verified": failed_sample["artifact_verified"],
                }
            if not all(item["rejected"] for item in oracle.values()):
                raise AssertionError(f"benchmark quality oracle accepted failure: {oracle}")

            for name, spec in scenario_specs.items():
                # One unreported warmup separates module/connection setup from
                # the warm-process samples compared between source trees.
                await run_once(user_id, workspace, spec, delay_ms / 1000)
                samples = [
                    await run_once(user_id, workspace, spec, delay_ms / 1000)
                    for _ in range(iterations)
                ]
                elapsed = [sample["elapsed_ms"] for sample in samples]
                output[name] = {
                    "median_ms": round(statistics.median(elapsed), 3),
                    "p95_ms_nearest_rank": round(nearest_rank(elapsed, 0.95), 3),
                    "min_ms": round(min(elapsed), 3),
                    "max_ms": round(max(elapsed), 3),
                    "underlying_calls": sorted({s["underlying_calls"] for s in samples}),
                    "logical_results": sorted({s["logical_results"] for s in samples}),
                    "quality_passed": all(s["quality_passed"] for s in samples),
                    "raw_samples": [
                        {
                            **sample,
                            "elapsed_ms": round(sample["elapsed_ms"], 3),
                        }
                        for sample in samples
                    ],
                }

            probe_runner = ar.AgentRunner(
                llm_service=ScriptedLLM([]),
                tool_executor=ToolExecutor(workspace=workspace),
            )
            routing = {}
            for language, message in {
                "english": "Search the latest AI news with sources",
                "persian": "آخرین اخبار هوش مصنوعی را جستجو کن و با منبع بده",
            }.items():
                intent = classify_query_intent(message)
                filtered = filter_tools_by_intent(probe_runner.tool_defs, intent)
                routing[language] = {
                    "intent": intent.category,
                    "allowed_tool_count": len(filtered),
                    "allowed_tool_json_bytes": len(json.dumps(filtered, default=str)),
                    "web_search_available": any(
                        (tool.get("name") or tool.get("function", {}).get("name"))
                        == "web_search" for tool in filtered
                    ),
                }
    finally:
        await drop_db()
        await engine.dispose()

    print(json.dumps({
        "source": str(SOURCE),
        "iterations": iterations,
        "warmup_samples_per_scenario": 1,
        "delay_ms_per_underlying_tool": delay_ms,
        "parallel_cap": 2,
        "percentile_method": "nearest-rank",
        "conditions": "warm process; real AgentRunner; controlled LLM/tools; no network",
        "quality_oracle": oracle,
        "quality_policy": (
            "Each logical tool_result must contain its expected successful fixture evidence; "
            "errors, missing results, wrong queries, or an absent/mismatched local artifact fail."
        ),
        "routing": routing,
        "scenarios": output,
    }, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--delay-ms", type=float, default=60.0)
    args = parser.parse_args()
    if args.iterations < 2:
        parser.error("--iterations must be at least 2")
    asyncio.run(main(args.iterations, args.delay_ms))
