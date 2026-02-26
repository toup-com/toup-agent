#!/usr/bin/env python3
"""Initialize database and create default user on startup"""
import asyncio
import os
from app.db import init_db, async_session_maker
from app.services.auth_service import create_user, get_user_by_email


async def init_startup():
    """Initialize database and create default user"""
    print("🧠 Initializing HexBrain database...")
    await init_db()
    print("✅ Database initialized")
    
    # Create default user if not exists
    async with async_session_maker() as db:
        email = os.getenv("DEFAULT_USER_EMAIL", "hex")
        password = os.getenv("DEFAULT_USER_PASSWORD", "Nariman123!")
        name = os.getenv("DEFAULT_USER_NAME", "Hex User")
        
        existing = await get_user_by_email(db, email)
        if not existing:
            user = await create_user(db, email, password, name)
            print(f"✅ Created user: {user.email}")
        else:
            print(f"ℹ️  User {email} already exists")


if __name__ == "__main__":
    asyncio.run(init_startup())
