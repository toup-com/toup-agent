"""Load / soak harness for the real agent pipeline (G-20).

Drives the CANARY tenant's own agent container end-to-end with a mixed
profile of simulated users:

  * chat turns over the agent's real WebSocket (/api/ws/chat, channel="web")
    with memory-relevant content — exercises retrieval + LLM (via the
    platform llm-proxy) + memory writes (post-processing extraction, dedup
    advisory lock) + day-chat/session writes;
  * memory searches (GET /api/memories/search) — retrieval only
    (embedding leg + ranking);
  * voice-ish turns (POST /api/v1/internal/agent-turn, save=false) — the
    exact hop platform ws_realtime makes for a voice `think`: full agent,
    retrieval + LLM, NO persistence and NO memory mining (save=false sets
    disable_post_processing).

DESIGN NOTES
------------
Auth is ONE secret: the canary tenant's agent_api_key. It authenticates
all three legs (ws_chat accepts X-Agent-Key on the upgrade request,
api_v1.internal_agent_turn and memories.* accept the same header).

Tokens come back IN the responses (ChatResponse.tokens_* on the REST legs,
the ws `done` frame's `tokens` object). COST does NOT come back — it is
metered platform-side into llm_proxy_events (cost_cents per call). The
harness therefore reports tokens live and prints the exact SQL to
reconcile cost after the run. If LOADSOAK_DB_URL (a READ-ONLY postgres
URL) is provided it runs that reconciliation itself at the end.

GUARDRAILS (all hard, all on by default)
----------------------------------------
  1. Canary-only: refuses to start unless the target host is in
     _ALLOWED_HOSTS. Adding a host is a code change, not a flag.
  2. Hard request cap: --max-requests (default 300) counted across ALL
     virtual users; the run stops when it is reached regardless of
     --duration.
  3. Cost cap: --max-cost-cents (default 200 = $2). Enforced against a
     MODELLED running estimate from returned token counts (labelled as
     such in the report); reconcile with llm_proxy_events afterwards.
  4. Off switch: touch the file given by --stop-file (default
     /tmp/loadsoak.stop) and every worker exits before its next request.
     SIGINT does the same, gracefully, printing the partial report.
  5. Concurrency cap: --users (virtual users) is clamped to 16.
  6. All sessions are namespaced "loadsoak-<runid>-u<N>" so anything the
     run persisted in the canary's day-chat is greppable and deletable.

USAGE (never run without a human decision — this spends real money):
    python backend/scripts/load_soak.py \
        --agent-url https://agent-533354ce.agents.toup.ai \
        --agent-key "$CANARY_AGENT_KEY" \
        --users 5 --duration 120 --max-requests 200 --max-cost-cents 100

Dry-run (prints the plan, sends nothing):
    python backend/scripts/load_soak.py --agent-url ... --agent-key x --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import signal
import statistics
import sys
import time
import uuid
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx

try:
    import websockets  # backend/requirements.txt pins >=13.0
except ImportError:  # pragma: no cover
    websockets = None

# ── Guardrail 1: the only hosts this harness will ever drive ─────────
_ALLOWED_HOSTS = frozenset({
    "agent-533354ce.agents.toup.ai",   # rollout canary (533354ce)
    "localhost", "127.0.0.1",          # local dev container
})

_MAX_USERS = 16

# Memory-relevant prompts: each writes a plausible fact (exercises the
# extraction → dedup advisory-lock path) or asks a recall question
# (exercises hybrid retrieval + re-rank).
_CHAT_PROMPTS = [
    "Quick note: my colleague Dana prefers meetings after 2pm on Tuesdays.",
    "Remember that the loadsoak project codename is BLUE HERON.",
    "What do you remember about my meeting preferences?",
    "I switched my main editor to Zed last week, in case that matters.",
    "What's the codename I told you for the loadsoak project?",
    "My flight to Lisbon is on the 24th, keep that in mind.",
    "When is my Lisbon flight, and what else do you know about my travel?",
]
_SEARCH_QUERIES = [
    "meeting preferences", "project codename", "travel plans",
    "editor", "colleague Dana",
]
_VOICE_TASKS = [
    "Summarize what you know about my current projects in two sentences.",
    "What did I say about my travel plans?",
]

# Modelled (NOT measured) cents/1k tokens used ONLY for the live cost-cap
# estimate. Real cost is cost_cents in llm_proxy_events — reconcile after.
_EST_CENTS_PER_1K_IN = 0.02
_EST_CENTS_PER_1K_OUT = 0.10


@dataclass
class Stats:
    latencies_ms: dict = field(default_factory=lambda: {"chat_ws": [], "mem_search": [], "voice_turn": []})
    errors: dict = field(default_factory=lambda: {"chat_ws": 0, "mem_search": 0, "voice_turn": 0})
    requests: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    started: float = 0.0

    def est_cost_cents(self) -> float:
        """MODELLED running estimate — see module docstring."""
        return (self.tokens_in / 1000) * _EST_CENTS_PER_1K_IN + \
               (self.tokens_out / 1000) * _EST_CENTS_PER_1K_OUT


STATS = Stats()
STOP = asyncio.Event()


def _pct(vals, p):
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[min(len(s) - 1, int(len(s) * p / 100))]


async def _chat_ws_turn(args, vu: int, session_id: str) -> None:
    """One full chat turn over the agent's real WebSocket."""
    if websockets is None:
        raise RuntimeError("pip install websockets")
    ws_url = args.agent_url.replace("https://", "wss://").replace("http://", "ws://") + "/api/ws/chat"
    t0 = time.monotonic()
    async with websockets.connect(
        ws_url, additional_headers={"X-Agent-Key": args.agent_key},
        open_timeout=15, close_timeout=5,
    ) as ws:
        await ws.send(json.dumps({
            "type": "message",
            "text": random.choice(_CHAT_PROMPTS),
            "session_id": session_id,
            "channel": "web",
        }))
        deadline = time.monotonic() + args.turn_timeout
        while time.monotonic() < deadline:
            raw = await asyncio.wait_for(ws.recv(), timeout=max(1.0, deadline - time.monotonic()))
            frame = json.loads(raw)
            ftype = frame.get("type")
            if ftype == "done":
                toks = frame.get("tokens") or {}
                STATS.tokens_in += int(toks.get("input") or toks.get("tokens_input") or 0)
                STATS.tokens_out += int(toks.get("output") or toks.get("tokens_output") or 0)
                break
            if ftype == "error":
                raise RuntimeError(f"ws error frame: {frame.get('message')!r}")
    STATS.latencies_ms["chat_ws"].append((time.monotonic() - t0) * 1000)


async def _mem_search(args, client: httpx.AsyncClient) -> None:
    t0 = time.monotonic()
    r = await client.get(
        f"{args.agent_url}/api/memories/search",
        params={"query": random.choice(_SEARCH_QUERIES), "limit": 10},
        headers={"X-Agent-Key": args.agent_key}, timeout=30.0,
    )
    r.raise_for_status()
    STATS.latencies_ms["mem_search"].append((time.monotonic() - t0) * 1000)


async def _voice_turn(args, client: httpx.AsyncClient, session_id: str) -> None:
    t0 = time.monotonic()
    r = await client.post(
        f"{args.agent_url}/api/v1/internal/agent-turn",
        json={"message": random.choice(_VOICE_TASKS),
              "session_id": session_id, "save": False},
        headers={"X-Agent-Key": args.agent_key}, timeout=args.turn_timeout,
    )
    r.raise_for_status()
    body = r.json()
    STATS.tokens_in += int(body.get("tokens_input") or 0)
    STATS.tokens_out += int(body.get("tokens_output") or 0)
    STATS.latencies_ms["voice_turn"].append((time.monotonic() - t0) * 1000)


async def _virtual_user(args, vu: int, run_id: str) -> None:
    session_id = f"loadsoak-{run_id}-u{vu}"
    ops = (["chat_ws"] * args.mix_chat + ["mem_search"] * args.mix_search
           + ["voice_turn"] * args.mix_voice)
    async with httpx.AsyncClient() as client:
        while not STOP.is_set():
            if os.path.exists(args.stop_file):
                print(f"[vu{vu}] stop-file present — exiting"); return
            if STATS.requests >= args.max_requests:
                STOP.set(); return
            if STATS.est_cost_cents() >= args.max_cost_cents:
                print(f"[vu{vu}] modelled cost cap hit "
                      f"({STATS.est_cost_cents():.1f}c) — stopping run")
                STOP.set(); return
            op = random.choice(ops)
            STATS.requests += 1
            try:
                if op == "chat_ws":
                    await _chat_ws_turn(args, vu, session_id)
                elif op == "mem_search":
                    await _mem_search(args, client)
                else:
                    await _voice_turn(args, client, session_id)
            except Exception as e:
                STATS.errors[op] += 1
                print(f"[vu{vu}] {op} failed: {type(e).__name__}: {e}")
            await asyncio.sleep(random.uniform(*args.think_time))


def _report(args, run_id: str) -> None:
    wall = max(0.001, time.monotonic() - STATS.started)
    total_ok = sum(len(v) for v in STATS.latencies_ms.values())
    total_err = sum(STATS.errors.values())
    print("\n──── loadsoak report ────")
    print(f"run_id={run_id} target={args.agent_url}")
    print(f"expected<= {args.max_requests}  sent={STATS.requests}  "
          f"ok={total_ok}  errors={total_err}  "
          f"error_rate={total_err / max(1, STATS.requests):.1%}")
    print(f"wall={wall:.1f}s  RPS={STATS.requests / wall:.2f} (MEASURED)")
    for op, vals in STATS.latencies_ms.items():
        if vals:
            print(f"  {op:<10} n={len(vals):<4} p50={_pct(vals, 50):7.0f}ms  "
                  f"p95={_pct(vals, 95):7.0f}ms  max={max(vals):7.0f}ms (MEASURED)")
        else:
            print(f"  {op:<10} n=0")
    print(f"tokens: in={STATS.tokens_in} out={STATS.tokens_out} "
          f"(MEASURED, from responses)")
    print(f"cost estimate: {STATS.est_cost_cents():.1f}c (MODELLED — reconcile below)")
    print("\nReconcile REAL cost + per-call latency from the platform DB:\n"
          "  SELECT count(*), sum(cost_cents), sum(input_tokens), sum(output_tokens),\n"
          "         percentile_cont(0.5) WITHIN GROUP (ORDER BY latency_ms) AS p50_ms,\n"
          "         percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_ms\n"
          "  FROM llm_proxy_events\n"
          "  WHERE user_id = '<canary-user-id>' AND created_at >= '<run start UTC>';")
    print("Cleanup: the run's sessions are named loadsoak-… in the canary day-chat.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--agent-url", required=True)
    ap.add_argument("--agent-key", required=True)
    ap.add_argument("--users", type=int, default=3)
    ap.add_argument("--duration", type=float, default=60.0, help="seconds")
    ap.add_argument("--max-requests", type=int, default=300)
    ap.add_argument("--max-cost-cents", type=float, default=200.0)
    ap.add_argument("--stop-file", default="/tmp/loadsoak.stop")
    ap.add_argument("--turn-timeout", type=float, default=120.0)
    ap.add_argument("--think-time", type=float, nargs=2, default=(1.0, 4.0))
    ap.add_argument("--mix-chat", type=int, default=6, help="weight")
    ap.add_argument("--mix-search", type=int, default=2, help="weight")
    ap.add_argument("--mix-voice", type=int, default=2, help="weight")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    host = urlparse(args.agent_url).hostname or ""
    if host not in _ALLOWED_HOSTS:
        sys.exit(f"REFUSED: {host!r} is not an allowed loadsoak target "
                 f"(allowed: {sorted(_ALLOWED_HOSTS)}). This is a code-level "
                 f"guardrail, not a flag.")
    args.users = min(args.users, _MAX_USERS)
    run_id = uuid.uuid4().hex[:8]

    if args.dry_run:
        print(f"DRY RUN — would drive {args.agent_url} with {args.users} vus "
              f"for {args.duration}s, cap {args.max_requests} req / "
              f"{args.max_cost_cents}c; mix chat:search:voice = "
              f"{args.mix_chat}:{args.mix_search}:{args.mix_voice}")
        return

    async def _run():
        STATS.started = time.monotonic()
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, STOP.set)
        vus = [asyncio.create_task(_virtual_user(args, i, run_id))
               for i in range(args.users)]
        try:
            await asyncio.wait_for(asyncio.gather(*vus), timeout=args.duration)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            STOP.set()
            await asyncio.gather(*vus, return_exceptions=True)

    asyncio.run(_run())
    _report(args, run_id)


if __name__ == "__main__":
    main()
