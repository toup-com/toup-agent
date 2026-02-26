"""
Seed script to populate the database with sample data for demo purposes.
"""

import asyncio
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db import init_db, async_session_maker
from app.services import create_user, get_user_by_email, get_memory_extractor, get_embedding_service
from app.services.memory_service import MemoryService
from app.schemas import MemoryCreate, MemoryCategory, MemoryType


# Sample conversations to seed
SAMPLE_CONVERSATIONS = [
    {
        "user": "Hi! My name is Alex and I'm a software engineer working at TechCorp.",
        "assistant": "Nice to meet you, Alex! It's great that you're a software engineer at TechCorp. What kind of projects are you working on there?",
    },
    {
        "user": "I'm currently building a machine learning pipeline for our recommendation system. I really enjoy working with Python and TensorFlow.",
        "assistant": "That sounds like an exciting project! Machine learning recommendation systems are fascinating. Python and TensorFlow are excellent choices for ML development. Are you working on collaborative filtering or content-based recommendations?",
    },
    {
        "user": "My favorite programming language is Python, but I also like TypeScript for frontend work.",
        "assistant": "Python and TypeScript are a great combination! Python's versatility for backend and ML work pairs well with TypeScript's type safety for frontend development. Do you use any particular frameworks with TypeScript?",
    },
    {
        "user": "I need to remember to submit my quarterly report by Friday. It's really important.",
        "assistant": "I'll help you remember that! You have an important deadline: submit your quarterly report by Friday. Would you like me to remind you a day before as well?",
    },
    {
        "user": "Yesterday I attended a conference on AI safety and met Dr. Sarah Chen, who's an expert in the field.",
        "assistant": "AI safety is such an important topic! Meeting Dr. Sarah Chen sounds valuable - she's likely doing groundbreaking work. What were some key takeaways from the conference?",
    },
    {
        "user": "I prefer working from my home office in the mornings, usually between 7 AM and noon. That's when I'm most productive.",
        "assistant": "Morning productivity hours are valuable! Working from 7 AM to noon from your home office sounds like a great routine. Many people find their focus peaks during those early hours before distractions build up.",
    },
    {
        "user": "I've been learning about vector databases lately. I saved a great research paper at ~/Documents/vector-db-research.pdf",
        "assistant": "Vector databases are increasingly important for AI applications! That's smart to save that research paper. Vector databases are crucial for similarity search, recommendation systems, and RAG applications. What aspects interest you most?",
    },
    {
        "user": "My colleague Mike from the data team is helping me with the database optimization. He's really knowledgeable about PostgreSQL.",
        "assistant": "Having a PostgreSQL expert like Mike on your team is invaluable! Database optimization can significantly impact your ML pipeline's performance. Are you looking at indexing strategies or query optimization?",
    },
    {
        "user": "I learned how to deploy models using Docker and Kubernetes last month. It was challenging but worth it.",
        "assistant": "Containerization skills are essential for ML engineers! Docker and Kubernetes form the foundation of modern ML deployment. The learning curve is steep but those skills will serve you well. Have you explored MLflow or similar tools for model tracking?",
    },
    {
        "user": "I hate working late at night. I really prefer to keep work-life balance.",
        "assistant": "Maintaining work-life balance is so important! Avoiding late-night work helps prevent burnout and keeps you fresh for your morning productivity hours. Setting boundaries is healthy and leads to better long-term output.",
    },
    {
        "user": "Next week I'm flying to San Francisco for a tech meetup about LLMs and prompt engineering.",
        "assistant": "San Francisco tech meetups are always exciting, especially for LLMs and prompt engineering! That's such a hot topic right now. Are you planning to network or present something at the meetup?",
    },
    {
        "user": "I've decided to use Supabase for my personal projects because it offers good free tier and has pgvector support.",
        "assistant": "Supabase is an excellent choice for personal projects! The generous free tier combined with pgvector support makes it perfect for AI-powered applications. It's great that you're thinking about vector search capabilities from the start.",
    },
]


async def seed_database():
    """Seed the database with sample data"""
    print("🌱 Starting database seed...")
    
    # Initialize database
    await init_db()
    print("✅ Database initialized")
    
    async with async_session_maker() as db:
        # Create or get demo user
        demo_email = "demo@hexbrain.local"
        demo_password = "demo123456"
        
        user = await get_user_by_email(db, demo_email)
        if not user:
            user = await create_user(db, demo_email, demo_password, "Demo User")
            print(f"✅ Created demo user: {demo_email}")
        else:
            print(f"ℹ️ Demo user already exists: {demo_email}")
        
        # Initialize services
        extractor = get_memory_extractor()
        embedding_service = get_embedding_service()
        memory_service = MemoryService(db)
        
        print("🧠 Processing sample conversations...")
        
        total_memories = 0
        for i, conv in enumerate(SAMPLE_CONVERSATIONS):
            print(f"  Processing conversation {i + 1}/{len(SAMPLE_CONVERSATIONS)}...")
            
            # Extract memories from conversation
            extracted = extractor.extract_memories(
                conv["user"],
                conv["assistant"]
            )
            
            # Store each memory
            for ext in extracted:
                memory_data = MemoryCreate(
                    content=ext.content,
                    summary=ext.summary,
                    category=ext.category,
                    memory_type=ext.memory_type,
                    importance=ext.importance,
                    confidence=ext.confidence,
                    tags=ext.tags,
                    metadata=ext.metadata,
                )
                
                await memory_service.create_memory(user.id, memory_data)
                total_memories += 1
        
        print(f"\n✅ Seeding complete!")
        print(f"   Total memories created: {total_memories}")
        print(f"   Demo login: {demo_email} / {demo_password}")


if __name__ == "__main__":
    asyncio.run(seed_database())
