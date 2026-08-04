"""Randomized soak.

Skipped unless MEMVERIFY_SOAK=<n> is set:

    python backend/scripts/memory_verify.py --soak 500

Conversations are GENERATED, not hand-written, so the run covers combinations
nobody sat down and thought of. Each generated conversation still carries
machine-checkable labels, because the generator builds it from a fact whose
sentinel it knows and a junk statement whose sentinel it knows. Recall and junk
are measured against those, exactly as in the scripted corpus.

Pass bar: zero recall misses, zero junk entries, zero pipeline errors.
"""

from __future__ import annotations

import asyncio
import os
import random
import time

import pytest

from .conftest import LABELED_USER_NAME, make_user, record_metric
from .pipeline import Marker as M, Scenario, Turn as T, run_scenario

SOAK_N = int(os.environ.get("MEMVERIFY_SOAK", "0"))
pytestmark = pytest.mark.skipif(
    SOAK_N <= 0, reason="set MEMVERIFY_SOAK=<n> (e.g. --soak 500) to run"
)

# ── Generators ───────────────────────────────────────────────────────────
# Each entry is (template, sentinel_builder). The sentinel is a rare token the
# memory must contain, so a match cannot come from an unrelated row.

FACT_TEMPLATES = [
    ("My dog is called {s} and he's a whippet.", "Noted — {s}."),
    ("My sister {s} lives in Montreal.", "Got it."),
    ("I work at a company called {s} as a data engineer.", "Understood."),
    ("I'm building a side project named {s}.", "Sounds interesting."),
    ("My landlord's name is {s}.", "Noted."),
    ("I'm allergic to {s}, it's serious.", "I'll keep that in mind."),
    ("My gym is called {s} and I go on Tuesdays.", "Noted."),
    ("I drive a car I nicknamed {s}.", "Ha, good name."),
    ("My cat {s} is fifteen years old.", "That's a senior cat."),
    ("I studied at {s} University.", "Good to know."),
    ("من یک پروژه دارم به اسم {s}.", "متوجه شدم."),
    ("My co-founder goes by {s}.", "Noted."),
    ("I keep my savings at a credit union called {s}.", "Understood."),
    ("My favourite restaurant is {s} downtown.", "Noted."),
    ("I've been learning a language called {s} for two years.", "Nice."),
]

# Junk templates carry their OWN sentinel pool, because the word type has to fit
# the sentence. Sharing one pool across all of them produced "The user is
# feeling Zanzibaria today" — not a sentence any human would say, and therefore
# not defensible ground truth. A soak has to simulate inputs real users produce,
# so the place templates get places and the mood templates get moods.
JUNK_TEMPLATES = [
    (
        "Also, what's the capital of {s}?",
        "That would be a city in {s}.",
        ["Zanzibaria", "Volterrino", "Quarnwick", "Draymoor", "Sableton"],
    ),
    (
        "Quick question — how does {s} work in general?",
        "{s} is a broad topic; the standard explanation is that it involves "
        "several stages, each with its own tradeoffs, and practitioners usually "
        "start with the simplest configuration before tuning.",
        ["photolithography", "reinsurance", "sous-vide cooking", "cold forging"],
    ),
    (
        "I'm feeling {s} today.",
        "Sorry to hear it — want me to keep this short?",
        ["wretched", "sluggish", "restless", "wired", "queasy", "drained"],
    ),
    (
        "I'm sitting in {s} right now.",
        "No problem, I'll keep it short.",
        ["the car", "the waiting room", "a taxi", "the back of a bus",
         "the airport lounge", "my kitchen"],
    ),
    (
        "thanks, {s}!",
        "Any time.",
        ["champion", "legend", "superstar"],
    ),
    (
        "Summarize: 'Ambrose {s} is the chief executive of Northwind Rail and "
        "has led the company since 2009, having previously run a shipping line.'",
        "Northwind Rail's chief executive is Ambrose {s}, in post since 2009.",
        ["Kolonaki", "Petrichorus", "Perplexivity", "Quarnwick"],
    ),
]

FACT_SENTINELS = [
    "Marbleweft", "Quillon", "Vantabrook", "Ferrowick", "Sundermere",
    "Halvard", "Tiberon", "Oakhelm", "Pellucid", "Grimsby", "Thornquist",
    "Ashgrove", "Verdania", "Kestrelford", "Bramblewood", "Cindervale",
    "Duskwater", "Emberline", "Fairholm", "Glimmerdon",
]


def _make_scenario(rng: random.Random, i: int) -> Scenario:
    fact_tpl, fact_reply = rng.choice(FACT_TEMPLATES)
    junk_tpl, junk_reply, junk_pool = rng.choice(JUNK_TEMPLATES)

    fact_s = f"{rng.choice(FACT_SENTINELS)}{rng.randrange(1000, 9999)}"
    junk_s = rng.choice(junk_pool)

    turns = [T(fact_tpl.format(s=fact_s), fact_reply.format(s=fact_s))]
    junk_turn = T(junk_tpl.format(s=junk_s), junk_reply.format(s=junk_s))
    # Order varies so the fact is sometimes buried after noise.
    if rng.random() < 0.5:
        turns.append(junk_turn)
    else:
        turns.insert(0, junk_turn)

    return Scenario(
        id=f"SOAK-{i:04d}",
        turns=turns,
        must_store=[M("fact", [fact_s.lower()])],
        must_not_store=[M("junk", [junk_s.lower()])],
    )


async def test_soak():
    from app.db.database import async_session_maker, engine

    rng = random.Random(int(os.environ.get("MEMVERIFY_SOAK_SEED", "20260801")))
    scenarios = [_make_scenario(rng, i) for i in range(SOAK_N)]

    limit = int(os.environ.get("MEMVERIFY_CONCURRENCY", "8"))
    sem = asyncio.Semaphore(limit)
    results, errors = [], []
    done = 0

    async def _one(sc: Scenario):
        nonlocal done
        async with sem:
            try:
                async with async_session_maker() as session:
                    uid = await make_user(
                        session,
                        name=LABELED_USER_NAME,
                        email=f"soak-{sc.id.lower()}-{rng.randrange(10**9)}@memverify.local",
                    )
                    results.append(await run_scenario(session, uid, sc))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{sc.id}: {type(exc).__name__}: {exc}")
            finally:
                done += 1
                if done % 50 == 0:
                    print(f"  soak {done}/{SOAK_N}", flush=True)

    started = time.perf_counter()
    await asyncio.gather(*(_one(sc) for sc in scenarios))
    elapsed = time.perf_counter() - started

    misses = [f"{r.scenario_id}:{m}" for r in results for m in r.missed]
    junk = [f"{r.scenario_id}:{j}" for r in results for j in r.junk]
    facts_total = sum(len(r.stored) + len(r.missed) for r in results)

    record_metric("soak_conversations", SOAK_N)
    record_metric("soak_completed", len(results))
    record_metric("soak_seconds", round(elapsed, 1))
    record_metric("soak_facts_labeled", facts_total)
    record_metric("soak_recall_misses", len(misses))
    record_metric("soak_junk_stored", len(junk))
    record_metric("soak_errors", len(errors))
    record_metric(
        "soak_recall_pct",
        round(100 * (facts_total - len(misses)) / facts_total, 2) if facts_total else 100.0,
    )

    assert not errors, f"{len(errors)} pipeline errors:\n" + "\n".join(errors[:10])
    assert not misses, f"{len(misses)} recall misses:\n" + "\n".join(misses[:20])
    assert not junk, f"{len(junk)} junk entries:\n" + "\n".join(junk[:20])
