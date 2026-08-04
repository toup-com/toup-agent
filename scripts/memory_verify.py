#!/usr/bin/env python3
"""One-command runner for the agent memory verification suite.

    python backend/scripts/memory_verify.py              # one clean run
    python backend/scripts/memory_verify.py --runs 3     # 3 consecutive clean runs
    python backend/scripts/memory_verify.py --soak 500   # randomized soak

What it does, in order:
  1. Refuses to touch anything that is not a disposable verification database.
  2. DROPs and re-CREATEs that database, with the `vector` and `pg_trgm`
     extensions, so every run starts from genuinely clean state.
  3. Exports RUN_MODE=agent + the Postgres DSN BEFORE pytest imports
     backend/tests/conftest.py (which uses os.environ.setdefault, so an
     exported value wins and the sqlite CI default is bypassed).
  4. Runs the suite and writes a machine-readable result file per run to
     artifacts/memverify/.

Requires: a local Postgres with the pgvector extension available, and
OPENAI_API_KEY (extraction and embeddings are NOT mocked).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
REPO = BACKEND.parent
ARTIFACTS = REPO / "artifacts" / "memverify"

DEFAULT_DB = os.environ.get("MEMVERIFY_DB", "toup_memtest")
DEFAULT_PGUSER = os.environ.get("MEMVERIFY_PGUSER", os.environ.get("USER", "postgres"))
DEFAULT_PGHOST = os.environ.get("MEMVERIFY_PGHOST", "localhost")
DEFAULT_PGPORT = os.environ.get("MEMVERIFY_PGPORT", "5432")
# psql picks the password up from PGPASSWORD; SQLAlchemy needs it in the DSN.
DEFAULT_PGPASSWORD = os.environ.get("MEMVERIFY_PGPASSWORD", os.environ.get("PGPASSWORD", ""))

# A production DSN must never reach this script. The database name has to
# advertise itself as disposable, because step 2 drops it.
_SAFE_DB_MARKERS = ("memtest", "memverify", "_test")


def _guard(dbname: str) -> None:
    if not any(m in dbname for m in _SAFE_DB_MARKERS):
        sys.exit(
            f"refusing to run against database {dbname!r}: the name must contain "
            f"one of {_SAFE_DB_MARKERS} so it is unmistakably disposable "
            "(this script DROPs it)"
        )
    for var in ("DATABASE_URL", "MEMVERIFY_PGHOST"):
        val = os.environ.get(var, "")
        if any(bad in val for bad in ("supabase", "pooler", "144.126.138.62", "railway")):
            sys.exit(f"refusing to run: {var} points at production ({val[:40]}...)")


def _psql(sql: str, *, db: str = "postgres") -> None:
    subprocess.run(
        ["psql", "-h", DEFAULT_PGHOST, "-p", DEFAULT_PGPORT, "-U", DEFAULT_PGUSER,
         "-d", db, "-v", "ON_ERROR_STOP=1", "-q", "-c", sql],
        check=True,
        stdout=subprocess.DEVNULL,
    )


def provision(dbname: str) -> None:
    _psql(
        f'SELECT pg_terminate_backend(pid) FROM pg_stat_activity '
        f"WHERE datname = '{dbname}' AND pid <> pg_backend_pid()"
    )
    _psql(f'DROP DATABASE IF EXISTS "{dbname}"')
    _psql(f'CREATE DATABASE "{dbname}"')
    _psql("CREATE EXTENSION IF NOT EXISTS vector", db=dbname)
    _psql("CREATE EXTENSION IF NOT EXISTS pg_trgm", db=dbname)


def build_env(dbname: str, run_id: str, results: Path) -> dict:
    env = dict(os.environ)
    auth = f"{DEFAULT_PGUSER}:{DEFAULT_PGPASSWORD}" if DEFAULT_PGPASSWORD else DEFAULT_PGUSER
    env.update(
        DATABASE_URL=f"postgresql+asyncpg://{auth}@{DEFAULT_PGHOST}:{DEFAULT_PGPORT}/{dbname}",
        RUN_MODE="agent",
        ENVIRONMENT="test",
        PYTHONPATH=str(BACKEND),
        # Production agent embedding config (agent images ship without
        # sentence-transformers, so `local` is not a faithful stand-in).
        EMBEDDING_PROVIDER="openai",
        EMBEDDING_MODEL="text-embedding-3-small",
        EMBEDDING_DIMENSION="1536",
        MEMVERIFY_RUN_ID=run_id,
        MEMVERIFY_RESULTS=str(results),
        SQLALCHEMY_WARN_20="0",
        # settings.debug drives SQLAlchemy echo; a full statement log per test
        # buries the assertions that matter.
        DEBUG="false",
    )
    env.setdefault("JWT_SECRET", "memverify-jwt-secret")
    env.setdefault("ENCRYPTION_KEY", "memverify-32-byte-encryption-key")
    return env


def load_dotenv_keys() -> None:
    """Pull the LLM keys from an env file if they are not already exported.

    Order: $MEMVERIFY_ENV_FILE, then backend/.env. The suite calls the real
    OpenAI API for both extraction and embeddings, so a working key is a hard
    requirement — there is no mock mode by design (a mocked extractor cannot
    measure extraction quality, which is what categories A and B exist to do).
    """
    candidates = []
    if os.environ.get("MEMVERIFY_ENV_FILE"):
        candidates.append(Path(os.environ["MEMVERIFY_ENV_FILE"]))
    candidates += [BACKEND / ".env", Path.home() / "toup-platform" / "backend" / ".env"]
    for candidate in candidates:
        if not candidate.exists():
            continue
        for line in candidate.read_text(errors="ignore").splitlines():
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY") and not os.environ.get(k):
                os.environ[k] = v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1, help="consecutive clean-state runs")
    ap.add_argument("--soak", type=int, default=0, help="randomized soak conversations")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("-k", dest="keyword", default=None, help="pytest -k filter")
    ap.add_argument("--maxfail", type=int, default=0)
    ap.add_argument("pytest_args", nargs="*", help="extra args passed to pytest")
    args = ap.parse_args()

    _guard(args.db)
    load_dotenv_keys()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY is required — this suite does not mock extraction")
    if not shutil.which("psql"):
        sys.exit("psql not found on PATH")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    overall = 0
    summaries = []

    for i in range(1, args.runs + 1):
        run_id = f"{stamp}-run{i}" if args.runs > 1 else stamp
        results = ARTIFACTS / f"{run_id}.json"
        print(f"\n=== memverify run {i}/{args.runs} — provisioning {args.db} ===", flush=True)
        provision(args.db)

        env = build_env(args.db, run_id, results)
        if args.soak:
            env["MEMVERIFY_SOAK"] = str(args.soak)

        cmd = [
            sys.executable, "-m", "pytest",
            str(BACKEND / "tests" / "memverify"),
            "-p", "no:cacheprovider",
            "-q", "--no-header", "-r", "fE",
        ]
        if args.keyword:
            cmd += ["-k", args.keyword]
        if args.maxfail:
            cmd += [f"--maxfail={args.maxfail}"]
        cmd += args.pytest_args

        rc = subprocess.run(cmd, cwd=str(BACKEND), env=env).returncode
        overall = overall or rc

        if results.exists():
            data = json.loads(results.read_text())
            t = data["totals"]
            print(
                f"--- run {i}: {t['passed']} passed, {t['failed']} failed, "
                f"{t['skipped']} skipped -> {results}",
                flush=True,
            )
            summaries.append({"run": run_id, "rc": rc, **t})
        else:
            summaries.append({"run": run_id, "rc": rc, "note": "no result file"})

        if rc != 0 and args.runs > 1:
            print(f"run {i} failed — stopping the consecutive-green attempt", flush=True)
            break

    index = ARTIFACTS / f"{stamp}-summary.json"
    index.write_text(json.dumps({"runs": summaries}, indent=2))
    print(f"\nsummary -> {index}")
    return overall


if __name__ == "__main__":
    raise SystemExit(main())
