"""Database initialization script"""

import asyncio
from app.db.database import init_db, engine
from app.db.models import Base


async def main():
    print("Creating database tables...")
    await init_db()
    print("Database initialized successfully!")


if __name__ == "__main__":
    asyncio.run(main())
