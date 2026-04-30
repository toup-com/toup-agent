"""
GET /api/models — canonical model registry endpoint.

Frontend dropdowns and any external tooling consume this instead of
hardcoding model strings. The endpoint is the single network-visible
source of truth; backend logic resolves through `model_resolver`,
frontend resolves through this endpoint.

Read-only and unauthenticated — model names are not sensitive. Browser
caching enabled via Cache-Control to keep load light.

Response shape:
    {
      "default": "claude-opus-4-7",
      "fallback": "gpt-5.4",
      "auto_builder": {
        "planner": "claude-opus-4-7",
        "builder": "claude-opus-4-7"
      },
      "anthropic": [
        {"id": "claude-opus-4-7", "label": "Claude Opus 4.7"},
        ...
      ],
      "openai": [
        {"id": "gpt-5.4",         "label": "GPT-5.4"},
        ...
      ]
    }

Lists come from `app.agent.model_providers.get_provider_registry()` —
the canonical "what's installable" source per audit §1.c.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(tags=["Models"])


# Display-label map. Frontend hands this to <option> rendering. The
# registry only stores model ids; this is the human-readable layer that
# was previously hardcoded across 5 frontend files.
_LABEL_MAP: dict[str, str] = {
    "claude-opus-4-7": "Claude Opus 4.7",
    "claude-opus-4-6": "Claude Opus 4.6",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "claude-sonnet-4-5-20250514": "Claude Sonnet 4.5",
    "claude-sonnet-4-20250514": "Claude Sonnet 4",
    "claude-3-5-haiku-20241022": "Claude 3.5 Haiku",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "gpt-5.4": "GPT-5.4",
    "gpt-5.2": "GPT-5.2",
    "gpt-5": "GPT-5",
    "gpt-4.1": "GPT-4.1",
    "gpt-4o": "GPT-4o",
    "gpt-4o-mini": "GPT-4o Mini",
    "gpt-4-turbo": "GPT-4 Turbo",
    "o1": "o1",
    "o1-mini": "o1-mini",
    "o3-mini": "o3-mini",
}


def _label_for(model_id: str) -> str:
    """Pretty-print a model id for UI dropdowns. Falls back to the id."""
    return _LABEL_MAP.get(model_id, model_id)


@router.get("/models")
async def list_models() -> JSONResponse:
    """Return the canonical model registry.

    No auth required, no per-user resolution — this advertises platform
    defaults + the catalogue of supported models. Tenants can still
    override `agent_model` / `app_builder_*_model` per-tenant; that
    happens at use-time in the agent's runtime, not here.
    """
    from app.services.model_resolver import (
        default_model,
        default_fallback_model,
        app_builder_planner_model,
        app_builder_builder_model,
    )
    from app.agent.model_providers import get_provider_registry

    registry = get_provider_registry()
    anthropic = registry.get_provider("anthropic")
    openai = registry.get_provider("openai")

    payload = {
        "default": default_model(),
        "fallback": default_fallback_model(),
        "auto_builder": {
            "planner": app_builder_planner_model(),
            "builder": app_builder_builder_model(),
        },
        "anthropic": [
            {"id": m, "label": _label_for(m)}
            for m in (anthropic.models if anthropic else [])
        ],
        "openai": [
            {"id": m, "label": _label_for(m)}
            for m in (openai.models if openai else [])
        ],
    }

    return JSONResponse(
        content=payload,
        # 5-minute browser cache. Endpoint cost is trivial but the
        # frontend hits it on every page mount; cache cuts that to
        # one request per session per browser tab.
        headers={"Cache-Control": "public, max-age=300"},
    )
