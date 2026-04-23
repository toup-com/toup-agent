"""Re-encrypt every streaming_credentials.password with the current primary key.

Usage (after setting PLATFORM_ENCRYPTION_KEY_PREVIOUS to the retiring key
and PLATFORM_ENCRYPTION_KEY to the new one):

    railway run --service platform-api python -m scripts.rotate_streaming_key

Idempotent: rows already on the primary are re-written identically. Once
every row returns True from `looks_encrypted` AND decrypts with the new
primary alone, the previous key can be removed from Railway env.
"""

import asyncio
import os

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.services.credential_crypto import rotate_token


async def main() -> None:
    url = os.environ["DATABASE_URL"]
    if url.endswith("?ssl"):
        url = url[:-5]
    # PgBouncer-transaction-mode-safe args (mirrors backend/app/db/database.py).
    engine = create_async_engine(
        url,
        connect_args={
            "statement_cache_size": 0,
            "prepared_statement_name_func": lambda: "",
        },
    )
    Session = async_sessionmaker(engine, expire_on_commit=False)

    # Lazy import so Alembic-only modules (services that depend on config) don't
    # complain if this script runs in a reduced env.
    from app.db.models import StreamingCredential

    rotated = 0
    total = 0
    async with Session() as db:
        rows = (await db.execute(select(StreamingCredential))).scalars().all()
        total = len(rows)
        for cred in rows:
            if cred.password is None:
                continue
            new_ct = rotate_token(cred.password)
            if new_ct == cred.password:
                # Already on primary; MultiFernet.rotate returns a new token
                # anyway, so this branch effectively never hits — left for
                # defensive readability.
                continue
            await db.execute(
                update(StreamingCredential)
                .where(StreamingCredential.id == cred.id)
                .values(password=new_ct)
            )
            rotated += 1
        await db.commit()

    print(f"rotate_streaming_key: rotated={rotated}/{total}")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
