"""
Demo mode script - simulates conversations and shows the brain updating live.
Run this after starting the backend server to see live memory ingestion.
"""

import asyncio
import httpx
import random
import sys

# Demo conversations that will be sent to the running server
DEMO_CONVERSATIONS = [
    ("My name is Jordan and I work as a product designer at StartupXYZ.", 
     "Nice to meet you, Jordan! Product design at a startup must be exciting. What kinds of products are you designing?"),
    
    ("I absolutely love coffee, especially Ethiopian single-origin beans.",
     "Ethiopian coffee is wonderful! The fruity and wine-like notes of Ethiopian beans are quite distinctive. Do you have a favorite roaster?"),
    
    ("I need to finish the design mockups for the mobile app by Wednesday.",
     "Got it! You have a deadline to complete the mobile app design mockups by Wednesday. That's an important deliverable!"),
    
    ("Last Tuesday I presented our new design system to the engineering team and got great feedback.",
     "That's fantastic! Presenting the design system to engineering shows great cross-functional collaboration. What aspects received the most positive feedback?"),
    
    ("I prefer using Figma for UI design and Notion for documentation.",
     "Figma and Notion are an excellent combination for design work! Figma's collaborative features pair well with Notion's flexibility for documentation."),
    
    ("My mentor Lisa from the UX team has been really helpful with my career growth.",
     "Having a mentor like Lisa is invaluable! Good mentorship can significantly accelerate your career development in UX/product design."),
    
    ("I learned about design tokens and component libraries from a workshop last month.",
     "Design tokens and component libraries are fundamental to scalable design systems! That workshop knowledge will definitely help with your design system work."),
    
    ("I saved our brand guidelines document at /Design/BrandGuidelines-v2.pdf",
     "I've noted the location of your brand guidelines: /Design/BrandGuidelines-v2.pdf. Having version 2 suggests you've been iterating on the brand!"),
    
    ("I don't like working on pixel-perfect designs without understanding the user context first.",
     "User context is indeed crucial! Understanding the 'why' before diving into pixel-perfect execution leads to more meaningful designs."),
    
    ("Next month I'm attending a design conference in Austin about AI-assisted design tools.",
     "A design conference in Austin sounds exciting! AI-assisted design tools are transforming the industry. Are you interested in any specific tools or approaches?"),
]


async def run_demo_simulation():
    """Run a demo simulation against the running backend"""
    base_url = "http://localhost:8000/api"
    
    print("🚀 Starting HexBrain Demo Simulation")
    print("=" * 50)
    print("Make sure the backend is running at http://localhost:8000")
    print("Open the frontend at http://localhost:5173 to watch the brain update!")
    print("=" * 50)
    print()
    
    async with httpx.AsyncClient() as client:
        # Login as demo user
        print("🔐 Logging in as demo user...")
        try:
            response = await client.post(f"{base_url}/auth/demo")
            response.raise_for_status()
            token_data = response.json()
            token = token_data["access_token"]
            print("✅ Logged in successfully!")
        except Exception as e:
            print(f"❌ Failed to login: {e}")
            print("   Make sure the backend is running and seeded.")
            return
        
        headers = {"Authorization": f"Bearer {token}"}
        
        # Process conversations one by one with delays
        print()
        print("🧠 Starting conversation simulation...")
        print("   Watch the 3D brain update in real-time!")
        print()
        
        for i, (user_msg, assistant_msg) in enumerate(DEMO_CONVERSATIONS):
            print(f"💬 Conversation {i + 1}/{len(DEMO_CONVERSATIONS)}")
            print(f"   User: {user_msg[:60]}...")
            
            # Send to ingestion endpoint
            try:
                response = await client.post(
                    f"{base_url}/ingest/message",
                    headers=headers,
                    json={
                        "user_message": user_msg,
                        "assistant_response": assistant_msg,
                        "extract_memories": True,
                    },
                    timeout=30.0,
                )
                response.raise_for_status()
                result = response.json()
                
                print(f"   ✅ Extracted {result['memories_extracted']} memories, {result['entities_extracted']} entities")
                
                # Show extracted memories
                for memory in result.get("memories", [])[:2]:
                    category = memory.get("category", "unknown")
                    content = memory.get("content", "")[:50]
                    print(f"      → [{category}] {content}...")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
            
            print()
            
            # Random delay to simulate real conversation
            delay = random.uniform(2, 4)
            print(f"   ⏳ Waiting {delay:.1f}s before next message...")
            await asyncio.sleep(delay)
        
        # Final stats
        print()
        print("=" * 50)
        print("📊 Fetching final brain stats...")
        
        try:
            response = await client.get(f"{base_url}/stats/regions", headers=headers)
            response.raise_for_status()
            stats = response.json()
            
            print(f"   Total memories: {stats['total_memories']}")
            print(f"   Total entities: {stats['total_entities']}")
            print(f"   Total connections: {stats['total_connections']}")
            print()
            print("   Memories by region:")
            for region in stats["regions"]:
                if region["count"] > 0:
                    bar = "█" * int(region["size"] * 20)
                    print(f"      {region['region']:15} {bar} ({region['count']})")
        except Exception as e:
            print(f"   ❌ Error fetching stats: {e}")
        
        print()
        print("✨ Demo simulation complete!")
        print("   Explore the brain at http://localhost:5173")


if __name__ == "__main__":
    asyncio.run(run_demo_simulation())
