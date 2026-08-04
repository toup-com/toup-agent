"""Database initialization script.

Lived at ``app/db/init_db.py`` until 2026-08-04, which was a landmine.
``app/db/__init__.py`` re-exports a FUNCTION named ``init_db`` (from
``app.db.database``), and importing a submodule binds it onto its package — so
anything that imported ``app.db.init_db`` would shadow that function for the
rest of the process, and the next ``from app.db import init_db; await
init_db()`` would raise ``TypeError: 'module' object is not callable``.

All three entrypoints (``agent_main``, ``platform_main``, ``app/main``) plus
``tests/conftest.py`` import that function at boot, so the blast radius was the
whole service. Nothing imported the module today — the collision was dormant,
not live — but it already cost a real debugging detour when a test-side scan
imported it, and a tripwire that buys nothing should not be left armed.

Run it as ``python -m app.scripts.init_db`` (it was ``python -m
app.db.init_db``). It sits beside its siblings now — ``init_startup.py``,
``seed_data.py`` — which is where it belonged anyway.
"""

import asyncio
from app.db.database import init_db, engine
from app.db.models import Base


async def main():
    print("Creating database tables...")
    await init_db()
    print("Database initialized successfully!")


if __name__ == "__main__":
    asyncio.run(main())
