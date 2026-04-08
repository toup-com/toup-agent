#!/usr/bin/env python3
"""
Manual wrapper for the DayChat backfill.

The same backfill logic runs automatically on agent startup via
app/services/backfill_day_chats.py. This script is for manual debugging
or emergency re-runs.

Usage:
    python scripts/backfill_day_chats_manual.py              # Execute
    python scripts/backfill_day_chats_manual.py --force       # Reset + re-run
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


async def main(force: bool):
    if force:
        os.environ["FORCE_DAY_CHAT_BACKFILL"] = "true"

    from app.db.database import async_session_maker
    from app.services.backfill_day_chats import run_backfill

    result = await run_backfill(async_session_maker)
    print(f"\nResult: {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual DayChat backfill")
    parser.add_argument("--force", action="store_true", help="Reset migration status and re-run")
    args = parser.parse_args()
    asyncio.run(main(force=args.force))
