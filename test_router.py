"""Smoke test for model_router."""
from app.services.model_router import classify_request

tests = [
    ("hi", "light"),
    ("thanks!", "light"),
    ("What is the capital of France?", "light"),
    ("Can you search the web for the latest news about AI and summarize it?", "medium"),
    ("Explain how transformers work in machine learning", "medium"),
    ("Implement a complete REST API with authentication, rate limiting, and database models. Include tests.", "heavy"),
    ("I need you to:\n1. Read all Python files\n2. Analyze the architecture\n3. Refactor the models\n4. Write tests\n5. Update docs", "heavy"),
    ("Write a comprehensive step-by-step guide to building a full-stack app", "heavy"),
]

print("=" * 80)
print("Model Router Smoke Test")
print("=" * 80)
for msg, expected_tier in tests:
    r = classify_request(msg)
    status = "PASS" if r.tier == expected_tier else "MISS"
    short_msg = msg[:60].replace("\n", " ")
    print(f"[{status}] '{short_msg}' => {r.tier}/{r.model} (expected {expected_tier})")
    print(f"       {r.reason}")
    print()
