"""Helper to import backfill without triggering app.services.__init__."""
import importlib.util
import sys
from pathlib import Path

_mod_path = Path(__file__).resolve().parent.parent / "app" / "services" / "backfill_day_chats.py"
spec = importlib.util.spec_from_file_location("backfill_day_chats", _mod_path)
_mod = importlib.util.module_from_spec(spec)
sys.modules["backfill_day_chats"] = _mod
spec.loader.exec_module(_mod)

run_backfill = _mod.run_backfill
MIGRATION_NAME = _mod.MIGRATION_NAME
