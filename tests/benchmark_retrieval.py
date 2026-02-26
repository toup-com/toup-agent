"""
Phase 8: Retrieval Benchmarking Suite
=====================================

Automated test suite that measures retrieval quality with known ground truth.

Metrics:
  - Precision@K: What fraction of retrieved memories are relevant?
  - Recall@K: What fraction of relevant memories are retrieved?
  - MRR (Mean Reciprocal Rank): How high is the first relevant result ranked?
  - nDCG@K: Normalized Discounted Cumulative Gain (ranking quality)

Strategies benchmarked:
  - vector: pgvector cosine similarity
  - keyword: tsvector full-text search + ILIKE fallback
  - graph: entity graph traversal via recursive CTE
  - temporal: recency-weighted time-window search
  - hybrid (all): RRF fusion of all strategies
  - hybrid + reranker: RRF + cross-encoder re-ranking

Usage:
  # Inside Docker container:
  python -m tests.benchmark_retrieval

  # Or via docker exec:
  docker exec hexbrain-backend python -m tests.benchmark_retrieval

  # Run specific benchmark:
  docker exec hexbrain-backend python -m tests.benchmark_retrieval --strategy vector
  docker exec hexbrain-backend python -m tests.benchmark_retrieval --strategy hybrid
"""

import asyncio
import json
import math
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

# ═══════════════════════════════════════════════════════════
# Ground Truth Memory IDs (from live DB, user nariman@toup.ai)
# User: aa57c9d4-9405-4a8a-a3a5-d9663a28ed8d
# ═══════════════════════════════════════════════════════════

USER_ID = "aa57c9d4-9405-4a8a-a3a5-d9663a28ed8d"

# Memory ID → short label (for readable output)
MEMORY_LABELS = {
    # === USER BRAIN ===
    # Identity
    "653f23d7-f0e0-4646-8406-21c602ee8c95": "nariman-identity-persian",
    "d1db9d51-c4e0-4c75-93de-c62fcb15bca0": "nariman-education-bsc",
    # Preferences
    "34360d82-e712-4c76-8c22-0a587a61fb2c": "nariman-prefers-direct",
    "3b72cc94-c1d1-4863-9ca5-7dff12938a00": "nariman-uses-claude-opus",
    # Projects
    "444dd6ec-57ca-46a9-94b8-dbbf5399bca1": "project-toup-travel",
    "aeaa3095-77b9-454b-9d4e-029d5537d8df": "project-hexbrain",
    "b3d6efec-73ca-4c8a-9e33-1c77e456a36b": "nariman-vps-hosting",
    # Goals
    "8a64dc65-a7e1-4459-9100-cb53ec0ce44e": "goal-mscac-uoft",
    # === AGENT BRAIN ===
    # Agent soul
    "bb52cecc-6794-481f-9477-6ac09299c9ef": "hex-identity-soul",
    # Agent decisions
    "e46a8fd4-39b9-4dee-90e9-fe223dabe731": "llm-routing-claude",
    "2a8f851c-2409-48a5-a380-d5317a4a5643": "architecture-lean-prompts",
    "26e5b712-9da0-4ea9-a7da-1ede47d7fa27": "oauth-claude-code",
    # Agent patterns
    "1896ae8b-34f6-4a81-8ad6-240b4061c429": "docker-compose-gotcha",
    "1a25b195-d5fa-4e0b-8a6d-bbecb39b861e": "disk-space-warning",
    "60c22ae6-e62c-4ae0-9b65-c78eeb642f21": "extraction-quality-issue",
    "86aefdc8-4782-4446-99e1-30cf12152502": "when-to-ask-vs-act",
    "67344244-9150-417e-aa35-b1242f81b7fc": "cost-awareness-opus",
    "7faa7a73-b609-4eaf-bddf-cacb6d95999e": "handle-nariman-direct",
    # Agent procedures
    "40985b06-efd7-45a5-bc51-34eadec88333": "github-repo-workflow",
    "64a9dafe-8534-4eb5-9702-6809b2e053be": "code-deployment-pattern",
    "f51094ab-064b-4db9-917d-f7df8c6ae701": "git-workflow-brain",
    "1a49fec4-f9f8-4b7c-a287-633d363ca41f": "memory-extraction-best",
    "43cb3f39-3d37-471d-8427-681dbde099f3": "vps-maintenance-tasks",
    "49c2873d-f4f5-457b-95c3-681b6565ddbb": "update-hexbrain-code",
    # Agent tools
    "1530a79d-f5be-4342-9513-0d42895e934d": "hex-tools-list",
    "345d4caf-37f6-464e-81f8-e666c328cb89": "vps-access-docker",
    "4c646a0d-4eeb-43aa-bbed-3b4c70d254c7": "hexbrain-architecture",
    # Agent goals
    "7a7ecf9a-764f-4b91-8eee-023fbdc3a234": "goal-priorities-mscac",
    # Agent tools (misc)
    "ed37932a-cac7-4089-87ab-f7284b3cc0d3": "telegram-formatting",
}

# ═══════════════════════════════════════════════════════════
# Benchmark Queries with Ground Truth
# ═══════════════════════════════════════════════════════════
# Each query has:
#   - query: the natural language question
#   - relevant_ids: set of memory IDs that are relevant to this query
#   - description: what we're testing
#   - category: which retrieval capability is being tested

BENCHMARK_QUERIES = [
    # ─── Identity Queries ───
    {
        "query": "What is my name?",
        "relevant_ids": {
            "653f23d7-f0e0-4646-8406-21c602ee8c95",  # nariman-identity-persian
            "d1db9d51-c4e0-4c75-93de-c62fcb15bca0",  # nariman-education-bsc
        },
        "description": "Basic identity recall — name, origin",
        "category": "identity",
    },
    {
        "query": "Where am I from?",
        "relevant_ids": {
            "653f23d7-f0e0-4646-8406-21c602ee8c95",  # Persian, born Feb 19 1998, Toronto
        },
        "description": "Origin / nationality",
        "category": "identity",
    },
    {
        "query": "Where do I live?",
        "relevant_ids": {
            "653f23d7-f0e0-4646-8406-21c602ee8c95",  # Toronto, Canada since Dec 2021
        },
        "description": "Current location",
        "category": "identity",
    },
    {
        "query": "What is my educational background?",
        "relevant_ids": {
            "d1db9d51-c4e0-4c75-93de-c62fcb15bca0",  # BSc Mech Eng, AI diploma Centennial
            "8a64dc65-a7e1-4459-9100-cb53ec0ce44e",  # MScAC at UofT
        },
        "description": "Education history",
        "category": "identity",
    },
    {
        "query": "When is my birthday?",
        "relevant_ids": {
            "653f23d7-f0e0-4646-8406-21c602ee8c95",  # born February 19, 1998
        },
        "description": "Date of birth recall",
        "category": "identity",
    },

    # ─── Project Queries ───
    {
        "query": "What projects am I working on?",
        "relevant_ids": {
            "444dd6ec-57ca-46a9-94b8-dbbf5399bca1",  # Toup travel platform
            "aeaa3095-77b9-454b-9d4e-029d5537d8df",  # HexBrain
            "b3d6efec-73ca-4c8a-9e33-1c77e456a36b",  # VPS hosting multiple projects
        },
        "description": "All projects overview",
        "category": "projects",
    },
    {
        "query": "Tell me about HexBrain",
        "relevant_ids": {
            "aeaa3095-77b9-454b-9d4e-029d5537d8df",  # HexBrain is Nariman's AI platform
            "4c646a0d-4eeb-43aa-bbed-3b4c70d254c7",  # HexBrain architecture
            "bb52cecc-6794-481f-9477-6ac09299c9ef",  # Hex identity (built on HexBrain)
        },
        "description": "HexBrain project details",
        "category": "projects",
    },
    {
        "query": "What is Toup?",
        "relevant_ids": {
            "444dd6ec-57ca-46a9-94b8-dbbf5399bca1",  # Toup multi-agent travel platform
        },
        "description": "Toup project recall",
        "category": "projects",
    },
    {
        "query": "What's running on the VPS?",
        "relevant_ids": {
            "b3d6efec-73ca-4c8a-9e33-1c77e456a36b",  # VPS hosting list
            "345d4caf-37f6-464e-81f8-e666c328cb89",  # VPS access + docker containers
            "4c646a0d-4eeb-43aa-bbed-3b4c70d254c7",  # HexBrain architecture on VPS
        },
        "description": "VPS / infrastructure recall",
        "category": "projects",
    },

    # ─── Goals Queries ───
    {
        "query": "What are my goals?",
        "relevant_ids": {
            "8a64dc65-a7e1-4459-9100-cb53ec0ce44e",  # MScAC at UofT
            "7a7ecf9a-764f-4b91-8eee-023fbdc3a234",  # Goal priorities (MScAC #1)
        },
        "description": "Goals and priorities",
        "category": "goals",
    },
    {
        "query": "Tell me about the MScAC program",
        "relevant_ids": {
            "8a64dc65-a7e1-4459-9100-cb53ec0ce44e",  # MScAC AI Concentration, UofT
            "7a7ecf9a-764f-4b91-8eee-023fbdc3a234",  # MScAC is #1 priority
        },
        "description": "MScAC details",
        "category": "goals",
    },

    # ─── Preferences Queries ───
    {
        "query": "How should you communicate with me?",
        "relevant_ids": {
            "34360d82-e712-4c76-8c22-0a587a61fb2c",  # Direct, concise, no fluff
            "7faa7a73-b609-4eaf-bddf-cacb6d95999e",  # Handle Nariman: direct, concise
        },
        "description": "Communication preferences",
        "category": "preferences",
    },
    {
        "query": "What LLM do I use?",
        "relevant_ids": {
            "3b72cc94-c1d1-4863-9ca5-7dff12938a00",  # Claude Opus 4.6 via Anthropic OAuth
            "e46a8fd4-39b9-4dee-90e9-fe223dabe731",  # LLM routing — Claude primary
        },
        "description": "LLM / model preferences",
        "category": "preferences",
    },

    # ─── Technical / Agent Queries ───
    {
        "query": "How do I deploy code to HexBrain?",
        "relevant_ids": {
            "64a9dafe-8534-4eb5-9702-6809b2e053be",  # Code deployment pattern: docker cp
            "49c2873d-f4f5-457b-95c3-681b6565ddbb",  # Update HexBrain code process
        },
        "description": "Deployment process recall",
        "category": "technical",
    },
    {
        "query": "What tools does Hex have?",
        "relevant_ids": {
            "1530a79d-f5be-4342-9513-0d42895e934d",  # My tools: Bash, Read/Write, etc.
            "345d4caf-37f6-464e-81f8-e666c328cb89",  # VPS access, docker containers
        },
        "description": "Agent tool capabilities",
        "category": "technical",
    },
    {
        "query": "How does the git workflow work?",
        "relevant_ids": {
            "40985b06-efd7-45a5-bc51-34eadec88333",  # GitHub repo, push with token
            "f51094ab-064b-4db9-917d-f7df8c6ae701",  # Git workflow for brain repo
        },
        "description": "Git workflow recall",
        "category": "technical",
    },
    {
        "query": "What are the Telegram formatting rules?",
        "relevant_ids": {
            "ed37932a-cac7-4089-87ab-f7284b3cc0d3",  # Telegram: no markdown tables, use HTML
        },
        "description": "Telegram formatting constraints",
        "category": "technical",
    },
    {
        "query": "How expensive is Claude?",
        "relevant_ids": {
            "67344244-9150-417e-aa35-b1242f81b7fc",  # Cost awareness: ~$15/M input
        },
        "description": "Cost awareness",
        "category": "technical",
    },

    # ─── Agent Knowledge Queries ───
    {
        "query": "Who are you?",
        "relevant_ids": {
            "bb52cecc-6794-481f-9477-6ac09299c9ef",  # I am Hex, AI executive assistant
        },
        "description": "Agent identity / soul",
        "category": "agent",
    },
    {
        "query": "What happened with disk space?",
        "relevant_ids": {
            "1a25b195-d5fa-4e0b-8a6d-bbecb39b861e",  # Disk space warning: 100% on Feb 11
            "43cb3f39-3d37-471d-8427-681dbde099f3",  # VPS maintenance: check disk space
        },
        "description": "Disk space incident recall",
        "category": "agent",
    },
    {
        "query": "What are the docker compose issues?",
        "relevant_ids": {
            "1896ae8b-34f6-4a81-8ad6-240b4061c429",  # Docker compose gotcha: env_file
        },
        "description": "Docker compose knowledge",
        "category": "agent",
    },

    # ─── Temporal Queries ───
    {
        "query": "What happened this month?",
        "relevant_ids": set(MEMORY_LABELS.keys()),  # All memories are from Feb 2026
        "description": "Temporal: this month (all memories qualify)",
        "category": "temporal",
    },
    {
        "query": "What happened recently?",
        "relevant_ids": set(MEMORY_LABELS.keys()),  # All within 7 days
        "description": "Temporal: recently (all within 7 days)",
        "category": "temporal",
    },

    # ─── Keyword-Specific Queries ───
    {
        "query": "PostgreSQL pgvector",
        "relevant_ids": {
            "4c646a0d-4eeb-43aa-bbed-3b4c70d254c7",  # HexBrain architecture: PostgreSQL + pgvector
            "345d4caf-37f6-464e-81f8-e666c328cb89",  # VPS access: hexbrain-db (pgvector)
        },
        "description": "Exact keyword match test",
        "category": "keyword",
    },
    {
        "query": "Nariman Iranian Persian",
        "relevant_ids": {
            "653f23d7-f0e0-4646-8406-21c602ee8c95",  # Nariman is Persian (Iranian)
        },
        "description": "Multi-keyword identity match",
        "category": "keyword",
    },
]


# ═══════════════════════════════════════════════════════════
# IR Metrics
# ═══════════════════════════════════════════════════════════

def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Fraction of top-k results that are relevant."""
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    return len(set(top_k) & relevant_ids) / len(top_k)


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Fraction of relevant results that appear in top-k."""
    if not relevant_ids:
        return 1.0  # No relevant docs = trivially satisfied
    top_k = set(retrieved_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def mrr(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """Mean Reciprocal Rank — 1/position of first relevant result."""
    for i, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain.
    
    Binary relevance: 1 if in relevant set, 0 otherwise.
    DCG@k = Σ rel_i / log2(i+1) for i=1..k
    IDCG@k = DCG of ideal ranking (all relevant first)
    nDCG@k = DCG@k / IDCG@k
    """
    if not relevant_ids or k <= 0:
        return 0.0
    
    top_k = retrieved_ids[:k]
    
    # DCG
    dcg = 0.0
    for i, rid in enumerate(top_k, start=1):
        rel = 1.0 if rid in relevant_ids else 0.0
        dcg += rel / math.log2(i + 1)
    
    # IDCG (ideal: all relevant docs first)
    n_relevant = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_relevant + 1))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def compute_metrics(retrieved_ids: List[str], relevant_ids: Set[str]) -> Dict[str, float]:
    """Compute all metrics for a single query."""
    return {
        "p@1": precision_at_k(retrieved_ids, relevant_ids, 1),
        "p@3": precision_at_k(retrieved_ids, relevant_ids, 3),
        "p@5": precision_at_k(retrieved_ids, relevant_ids, 5),
        "r@5": recall_at_k(retrieved_ids, relevant_ids, 5),
        "r@10": recall_at_k(retrieved_ids, relevant_ids, 10),
        "r@15": recall_at_k(retrieved_ids, relevant_ids, 15),
        "mrr": mrr(retrieved_ids, relevant_ids),
        "ndcg@5": ndcg_at_k(retrieved_ids, relevant_ids, 5),
        "ndcg@10": ndcg_at_k(retrieved_ids, relevant_ids, 10),
    }


# ═══════════════════════════════════════════════════════════
# Benchmark Runner
# ═══════════════════════════════════════════════════════════

async def run_single_query(
    svc,
    user_id: str,
    query: str,
    relevant_ids: Set[str],
    strategy: str = "hybrid",
    limit: int = 15,
) -> Dict[str, Any]:
    """Run a single benchmark query and return metrics."""
    
    t0 = time.time()
    
    if strategy == "hybrid":
        results = await svc.hybrid_search(
            user_id=user_id,
            query=query,
            limit=limit,
            strategies=["vector", "keyword", "graph"],
        )
    elif strategy == "vector":
        results = await svc.hybrid_search(
            user_id=user_id,
            query=query,
            limit=limit,
            strategies=["vector"],
        )
    elif strategy == "keyword":
        results = await svc.hybrid_search(
            user_id=user_id,
            query=query,
            limit=limit,
            strategies=["keyword"],
        )
    elif strategy == "graph":
        results = await svc.hybrid_search(
            user_id=user_id,
            query=query,
            limit=limit,
            strategies=["graph"],
        )
    elif strategy == "temporal":
        results = await svc.hybrid_search(
            user_id=user_id,
            query=query,
            limit=limit,
            strategies=["vector"],  # Temporal auto-activates
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    elapsed = time.time() - t0
    retrieved_ids = [r["id"] for r in results]
    
    metrics = compute_metrics(retrieved_ids, relevant_ids)
    metrics["latency_ms"] = elapsed * 1000
    metrics["n_retrieved"] = len(retrieved_ids)
    
    return {
        "metrics": metrics,
        "retrieved_ids": retrieved_ids,
        "results": results,
    }


async def run_benchmark(
    strategy: str = "hybrid",
    queries: Optional[List[dict]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the full benchmark suite for a given strategy.
    Returns aggregate metrics.
    """
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    from app.services.memory_service import MemoryService

    DATABASE_URL = "postgresql+asyncpg://hexbrain:hexbrain_secret@hexbrain-db:5432/hexbrain"
    engine = create_async_engine(DATABASE_URL)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    if queries is None:
        queries = BENCHMARK_QUERIES

    all_metrics = []
    query_results = []

    async with async_session() as db:
        svc = MemoryService(db)

        for i, q in enumerate(queries):
            result = await run_single_query(
                svc=svc,
                user_id=USER_ID,
                query=q["query"],
                relevant_ids=q["relevant_ids"],
                strategy=strategy,
                limit=15,
            )

            metrics = result["metrics"]
            all_metrics.append(metrics)

            # Check which relevant IDs were found
            found = set(result["retrieved_ids"]) & q["relevant_ids"]
            missed = q["relevant_ids"] - set(result["retrieved_ids"])

            query_results.append({
                "query": q["query"],
                "description": q["description"],
                "category": q["category"],
                "metrics": metrics,
                "found": [MEMORY_LABELS.get(mid, mid[:8]) for mid in found],
                "missed": [MEMORY_LABELS.get(mid, mid[:8]) for mid in missed],
            })

            if verbose:
                mrr_val = metrics["mrr"]
                p5 = metrics["p@5"]
                r5 = metrics["r@5"]
                latency = metrics["latency_ms"]
                status = "✅" if mrr_val >= 0.5 else "⚠️" if mrr_val > 0 else "❌"
                print(f"  {status} Q{i+1:02d}: MRR={mrr_val:.3f}  P@5={p5:.3f}  "
                      f"R@5={r5:.3f}  {latency:6.0f}ms | {q['query'][:50]}")
                if missed:
                    missed_labels = [MEMORY_LABELS.get(m, m[:8]) for m in missed]
                    print(f"       MISSED: {', '.join(missed_labels)}")

    await engine.dispose()

    # Aggregate metrics
    n = len(all_metrics)
    aggregate = {}
    for key in all_metrics[0]:
        aggregate[key] = sum(m[key] for m in all_metrics) / n

    # Per-category breakdown
    categories = set(q["category"] for q in queries)
    category_metrics = {}
    for cat in categories:
        cat_indices = [i for i, q in enumerate(queries) if q["category"] == cat]
        if cat_indices:
            cat_agg = {}
            for key in all_metrics[0]:
                cat_agg[key] = sum(all_metrics[i][key] for i in cat_indices) / len(cat_indices)
            category_metrics[cat] = cat_agg

    return {
        "strategy": strategy,
        "n_queries": n,
        "aggregate": aggregate,
        "category_metrics": category_metrics,
        "query_results": query_results,
    }


def print_report(result: Dict[str, Any]):
    """Print a formatted benchmark report."""
    agg = result["aggregate"]
    strategy = result["strategy"]
    n = result["n_queries"]
    
    print()
    print("=" * 78)
    print(f"  RETRIEVAL BENCHMARK REPORT — Strategy: {strategy.upper()}")
    print(f"  {n} queries | {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 78)
    print()
    
    # Overall metrics
    print("┌─────────────────────────────────────────────────────────────┐")
    print("│                    AGGREGATE METRICS                       │")
    print("├────────────┬──────────┬──────────┬──────────┬──────────────┤")
    print("│  Metric    │  Value   │  Target  │  Status  │  Notes       │")
    print("├────────────┼──────────┼──────────┼──────────┼──────────────┤")
    
    targets = {
        "p@1": (0.70, "First result relevant"),
        "p@3": (0.60, "Top 3 relevant"),
        "p@5": (0.85, "Blueprint target"),
        "r@5": (0.60, "Relevant in top 5"),
        "r@10": (0.70, "Relevant in top 10"),
        "r@15": (0.80, "Blueprint target"),
        "mrr": (0.75, "Blueprint target"),
        "ndcg@5": (0.70, "Ranking quality"),
        "ndcg@10": (0.75, "Ranking quality"),
    }
    
    for metric, (target, note) in targets.items():
        val = agg.get(metric, 0)
        status = "✅ PASS" if val >= target else "⚠️ CLOSE" if val >= target * 0.8 else "❌ FAIL"
        print(f"│  {metric:<10s}│  {val:6.3f}  │  {target:6.3f}  │ {status:8s} │  {note:<12s} │")
    
    lat = agg.get("latency_ms", 0)
    lat_status = "✅ PASS" if lat < 500 else "⚠️ SLOW" if lat < 1000 else "❌ FAIL"
    print(f"│  {'latency':<10s}│  {lat:5.0f}ms │  <500ms  │ {lat_status:8s} │  {'Avg query':12s} │")
    
    print("└────────────┴──────────┴──────────┴──────────┴──────────────┘")
    
    # Per-category breakdown
    print()
    print("  PER-CATEGORY BREAKDOWN:")
    print(f"  {'Category':<15s} {'P@5':>6s} {'R@5':>6s} {'R@15':>6s} {'MRR':>6s} {'nDCG@5':>7s} {'Latency':>8s}")
    print(f"  {'─'*15} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*7} {'─'*8}")
    
    for cat, cm in sorted(result["category_metrics"].items()):
        print(f"  {cat:<15s} {cm['p@5']:6.3f} {cm['r@5']:6.3f} {cm['r@15']:6.3f} "
              f"{cm['mrr']:6.3f} {cm['ndcg@5']:7.3f} {cm['latency_ms']:7.0f}ms")
    
    # Failed queries
    failures = [qr for qr in result["query_results"] if qr["metrics"]["mrr"] == 0]
    if failures:
        print()
        print(f"  ❌ FAILED QUERIES ({len(failures)}):")
        for f in failures:
            print(f"     • {f['query']}")
            print(f"       Missed: {', '.join(f['missed'])}")
    
    print()


async def benchmark_all_strategies():
    """Run benchmarks for all strategies and compare."""
    strategies = ["vector", "keyword", "hybrid"]
    results = {}
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"  Running benchmark: {strategy.upper()}")
        print(f"{'='*60}")
        result = await run_benchmark(strategy=strategy, verbose=True)
        results[strategy] = result
    
    # Comparison table
    print("\n" + "=" * 78)
    print("  STRATEGY COMPARISON")
    print("=" * 78)
    print(f"\n  {'Strategy':<20s} {'P@1':>6s} {'P@5':>6s} {'R@5':>6s} {'R@15':>6s} "
          f"{'MRR':>6s} {'nDCG@5':>7s} {'Latency':>8s}")
    print(f"  {'─'*20} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*6} {'─'*7} {'─'*8}")
    
    for strategy, result in results.items():
        agg = result["aggregate"]
        print(f"  {strategy:<20s} {agg['p@1']:6.3f} {agg['p@5']:6.3f} {agg['r@5']:6.3f} "
              f"{agg['r@15']:6.3f} {agg['mrr']:6.3f} {agg['ndcg@5']:7.3f} {agg['latency_ms']:7.0f}ms")
    
    # Winner per metric
    print()
    metric_names = ["p@1", "p@5", "r@5", "r@15", "mrr", "ndcg@5"]
    for metric in metric_names:
        best_strategy = max(results.keys(), key=lambda s: results[s]["aggregate"][metric])
        best_val = results[best_strategy]["aggregate"][metric]
        print(f"  Best {metric:>7s}: {best_strategy:<20s} ({best_val:.3f})")
    
    print()
    return results


# ═══════════════════════════════════════════════════════════
# Self-Test for Metrics (can run without DB)
# ═══════════════════════════════════════════════════════════

def test_metrics():
    """Unit tests for IR metric functions."""
    print("Testing IR metrics...")
    
    relevant = {"a", "b", "c"}
    retrieved = ["a", "d", "b", "e", "c"]
    
    # Precision@1 = 1/1 (a is relevant)
    assert precision_at_k(retrieved, relevant, 1) == 1.0
    # Precision@3 = 2/3 (a, b relevant out of a,d,b)
    assert abs(precision_at_k(retrieved, relevant, 3) - 2/3) < 1e-9
    # Precision@5 = 3/5
    assert abs(precision_at_k(retrieved, relevant, 5) - 3/5) < 1e-9
    
    # Recall@3 = 2/3
    assert abs(recall_at_k(retrieved, relevant, 3) - 2/3) < 1e-9
    # Recall@5 = 3/3 = 1.0
    assert recall_at_k(retrieved, relevant, 5) == 1.0
    
    # MRR = 1/1 (first result is relevant)
    assert mrr(retrieved, relevant) == 1.0
    # MRR when first relevant is at position 2
    assert mrr(["x", "a", "b"], relevant) == 0.5
    # MRR when nothing found
    assert mrr(["x", "y"], relevant) == 0.0
    
    # nDCG@5 — with all 3 relevant in top 5
    ndcg_val = ndcg_at_k(retrieved, relevant, 5)
    assert 0 < ndcg_val <= 1.0
    
    # Perfect ranking should give nDCG = 1.0
    perfect = ["a", "b", "c", "x", "y"]
    assert abs(ndcg_at_k(perfect, relevant, 5) - 1.0) < 1e-9
    
    # Empty cases
    assert precision_at_k([], relevant, 5) == 0.0
    assert recall_at_k([], relevant, 5) == 0.0
    assert mrr([], relevant) == 0.0
    assert recall_at_k(retrieved, set(), 5) == 1.0
    
    print("✅ All metric tests passed!")


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="HexBrain Retrieval Benchmark Suite")
    parser.add_argument("--strategy", type=str, default=None,
                        choices=["vector", "keyword", "graph", "temporal", "hybrid", "all"],
                        help="Strategy to benchmark (default: all)")
    parser.add_argument("--self-test", action="store_true",
                        help="Run metric self-tests (no DB needed)")
    parser.add_argument("--quiet", action="store_true",
                        help="Only show aggregate results")
    parser.add_argument("--json", action="store_true",
                        help="Output results as JSON")
    args = parser.parse_args()
    
    if args.self_test:
        test_metrics()
        sys.exit(0)
    
    if args.strategy and args.strategy != "all":
        result = asyncio.run(run_benchmark(
            strategy=args.strategy,
            verbose=not args.quiet,
        ))
        if args.json:
            # Serialize, removing non-serializable items
            output = {
                "strategy": result["strategy"],
                "n_queries": result["n_queries"],
                "aggregate": result["aggregate"],
                "category_metrics": result["category_metrics"],
            }
            print(json.dumps(output, indent=2))
        else:
            print_report(result)
    else:
        results = asyncio.run(benchmark_all_strategies())
        if args.json:
            output = {}
            for strategy, result in results.items():
                output[strategy] = {
                    "aggregate": result["aggregate"],
                    "category_metrics": result["category_metrics"],
                }
            print(json.dumps(output, indent=2))
