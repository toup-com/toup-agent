"""
Validate LLM provider API keys by making lightweight real API calls.
"""
import httpx
import logging

logger = logging.getLogger(__name__)

TIMEOUT = 10.0

PROVIDERS = {
    "openai", "anthropic", "google", "mistral", "groq", "xai", "deepseek",
}


async def validate_key(provider: str, api_key: str) -> dict:
    """
    Validate an API key for the given provider.
    Returns {"valid": True} or {"valid": False, "error": "..."}.
    """
    if provider not in PROVIDERS:
        return {"valid": False, "error": f"Unknown provider: {provider}"}

    if not api_key or not api_key.strip():
        return {"valid": False, "error": "API key is empty"}

    api_key = api_key.strip()

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            if provider == "openai":
                r = await client.get(
                    "https://api.openai.com/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
            elif provider == "anthropic":
                # OAuth tokens (sk-ant-oat*) use Bearer auth; regular keys use x-api-key
                is_oauth = api_key.startswith("sk-ant-oat")
                headers = {
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                }
                if is_oauth:
                    headers["Authorization"] = f"Bearer {api_key}"
                else:
                    headers["x-api-key"] = api_key
                r = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json={
                        "model": "claude-haiku-4-5-20251001",
                        "max_tokens": 1,
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                )
            elif provider == "google":
                r = await client.get(
                    f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
                )
            elif provider == "mistral":
                r = await client.get(
                    "https://api.mistral.ai/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
            elif provider == "groq":
                r = await client.get(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
            elif provider == "xai":
                r = await client.get(
                    "https://api.x.ai/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
            elif provider == "deepseek":
                r = await client.get(
                    "https://api.deepseek.com/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
            else:
                return {"valid": False, "error": f"Unsupported provider: {provider}"}

            if r.status_code == 200:
                return {"valid": True}

            # Anthropic returns 200 for valid keys even with minimal request
            # but also might return other success codes
            if provider == "anthropic" and r.status_code in (200, 201):
                return {"valid": True}

            # Common error patterns
            if r.status_code == 401:
                return {"valid": False, "error": "Invalid API key"}
            if r.status_code == 403:
                return {"valid": False, "error": "API key lacks required permissions"}
            if r.status_code == 429:
                # Rate limited but key is valid
                return {"valid": True}

            return {"valid": False, "error": f"API returned status {r.status_code}"}

    except httpx.TimeoutException:
        return {"valid": False, "error": "Request timed out — try again"}
    except httpx.ConnectError:
        return {"valid": False, "error": "Could not connect to provider API"}
    except Exception as e:
        logger.exception(f"Key validation error for {provider}")
        return {"valid": False, "error": str(e)}
