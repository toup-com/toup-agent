"""
Model Providers — Custom model provider configuration.

Allows configuring custom model providers with custom base URLs,
API keys, and model mappings. Supports OpenAI-compatible APIs
like Together, Groq, Mistral, local LLMs (LM Studio, Ollama), etc.

Usage:
    from app.agent.model_providers import get_provider_registry

    reg = get_provider_registry()
    reg.register_provider(ProviderConfig(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        models=["llama-3.3-70b", "mixtral-8x7b"],
    ))

    provider = reg.get_provider_for_model("llama-3.3-70b")
"""

import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ProviderType(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENAI_COMPATIBLE = "openai_compatible"
    LOCAL = "local"
    CUSTOM = "custom"


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    provider: str
    max_tokens: int = 8192
    context_window: int = 128000
    supports_vision: bool = False
    supports_tools: bool = True
    supports_streaming: bool = True
    cost_per_input_token: float = 0.0
    cost_per_output_token: float = 0.0
    aliases: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.provider,
            "max_tokens": self.max_tokens,
            "context_window": self.context_window,
            "supports_vision": self.supports_vision,
            "supports_tools": self.supports_tools,
            "cost_per_input_token": self.cost_per_input_token,
            "cost_per_output_token": self.cost_per_output_token,
        }


@dataclass
class ProviderConfig:
    """Configuration for a model provider."""
    name: str
    provider_type: ProviderType = ProviderType.OPENAI_COMPATIBLE
    base_url: str = ""
    api_key_env: str = ""
    api_key: str = ""  # Direct key (less preferred)
    models: List[str] = field(default_factory=list)
    model_configs: Dict[str, ModelConfig] = field(default_factory=dict)
    default_model: str = ""
    enabled: bool = True
    priority: int = 100  # Lower = higher priority
    headers: Dict[str, str] = field(default_factory=dict)
    registered_at: float = 0.0

    def __post_init__(self):
        if self.registered_at == 0.0:
            self.registered_at = time.time()

    def get_api_key(self) -> str:
        """Get the API key, preferring env var."""
        if self.api_key_env:
            return os.environ.get(self.api_key_env, self.api_key)
        return self.api_key

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "provider_type": self.provider_type.value,
            "base_url": self.base_url,
            "models": self.models,
            "default_model": self.default_model,
            "enabled": self.enabled,
            "priority": self.priority,
            "has_api_key": bool(self.get_api_key()),
        }


class ModelProviderRegistry:
    """
    Registry for model providers and their models.

    Built-in providers (OpenAI, Anthropic) are registered by default.
    Custom providers can be added for OpenAI-compatible APIs.
    """

    def __init__(self):
        self._providers: Dict[str, ProviderConfig] = {}
        self._model_map: Dict[str, str] = {}  # model_name → provider_name
        self._register_builtins()

    def _register_builtins(self):
        """Register built-in providers."""
        self.register_provider(ProviderConfig(
            name="openai",
            provider_type=ProviderType.OPENAI,
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            models=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1", "o1-mini", "o3-mini"],
            default_model="gpt-4o",
            priority=10,
        ))

        self.register_provider(ProviderConfig(
            name="anthropic",
            provider_type=ProviderType.ANTHROPIC,
            base_url="https://api.anthropic.com",
            api_key_env="ANTHROPIC_API_KEY",
            models=["claude-sonnet-4-20250514", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"],
            default_model="claude-sonnet-4-20250514",
            priority=20,
        ))

    def register_provider(self, config: ProviderConfig) -> ProviderConfig:
        """Register a model provider."""
        self._providers[config.name] = config
        for model in config.models:
            self._model_map[model] = config.name
        logger.info(f"[PROVIDERS] Registered {config.name} ({len(config.models)} models)")
        return config

    def unregister_provider(self, name: str) -> bool:
        """Remove a provider."""
        provider = self._providers.pop(name, None)
        if not provider:
            return False
        for model in provider.models:
            self._model_map.pop(model, None)
        return True

    def get_provider(self, name: str) -> Optional[ProviderConfig]:
        """Get provider by name."""
        return self._providers.get(name)

    def get_provider_for_model(self, model_name: str) -> Optional[ProviderConfig]:
        """Find which provider serves a model."""
        provider_name = self._model_map.get(model_name)
        if provider_name:
            return self._providers.get(provider_name)

        # Check aliases in model configs
        for provider in self._providers.values():
            for mc in provider.model_configs.values():
                if model_name in mc.aliases:
                    return provider

        return None

    def resolve_model(self, model_name: str) -> Dict[str, Any]:
        """
        Resolve a model name to provider + config.

        Returns base_url, api_key, headers needed for API call.
        """
        provider = self.get_provider_for_model(model_name)
        if not provider:
            return {"error": f"No provider found for model: {model_name}"}

        if not provider.enabled:
            return {"error": f"Provider {provider.name} is disabled"}

        return {
            "provider": provider.name,
            "provider_type": provider.provider_type.value,
            "model": model_name,
            "base_url": provider.base_url,
            "api_key": provider.get_api_key(),
            "headers": provider.headers,
        }

    def list_providers(self, enabled_only: bool = False) -> List[Dict[str, Any]]:
        """List all providers."""
        providers = list(self._providers.values())
        if enabled_only:
            providers = [p for p in providers if p.enabled]
        providers.sort(key=lambda p: p.priority)
        return [p.to_dict() for p in providers]

    def list_models(self, provider_name: Optional[str] = None) -> List[str]:
        """List all available models."""
        if provider_name:
            provider = self._providers.get(provider_name)
            return provider.models if provider else []
        return list(self._model_map.keys())

    def add_model(
        self,
        provider_name: str,
        model_name: str,
        *,
        max_tokens: int = 8192,
        context_window: int = 128000,
        cost_input: float = 0.0,
        cost_output: float = 0.0,
    ) -> bool:
        """Add a model to a provider."""
        provider = self._providers.get(provider_name)
        if not provider:
            return False

        if model_name not in provider.models:
            provider.models.append(model_name)
        self._model_map[model_name] = provider_name

        config = ModelConfig(
            name=model_name,
            provider=provider_name,
            max_tokens=max_tokens,
            context_window=context_window,
            cost_per_input_token=cost_input,
            cost_per_output_token=cost_output,
        )
        provider.model_configs[model_name] = config
        return True

    def enable_provider(self, name: str) -> bool:
        """Enable a provider."""
        p = self._providers.get(name)
        if not p:
            return False
        p.enabled = True
        return True

    def disable_provider(self, name: str) -> bool:
        """Disable a provider."""
        p = self._providers.get(name)
        if not p:
            return False
        p.enabled = False
        return True

    def stats(self) -> Dict[str, Any]:
        """Get provider statistics."""
        return {
            "total_providers": len(self._providers),
            "enabled_providers": sum(1 for p in self._providers.values() if p.enabled),
            "total_models": len(self._model_map),
            "providers": {
                name: {
                    "models": len(p.models),
                    "enabled": p.enabled,
                    "has_key": bool(p.get_api_key()),
                }
                for name, p in self._providers.items()
            },
        }


# ── Singleton ────────────────────────────────────────────
_registry: Optional[ModelProviderRegistry] = None


def get_provider_registry() -> ModelProviderRegistry:
    """Get the global model provider registry."""
    global _registry
    if _registry is None:
        _registry = ModelProviderRegistry()
    return _registry
