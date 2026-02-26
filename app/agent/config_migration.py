"""
Config Migration — Auto-migrate old configuration formats.

When config format changes between versions, migration rules
automatically transform old keys/values to the new format.
Keeps a version stamp and applies migrations sequentially.

Usage:
    from app.agent.config_migration import get_migration_manager

    mgr = get_migration_manager()
    mgr.register_migration("1.0", "2.0", migrate_v1_to_v2)
    new_config = mgr.migrate(old_config, from_version="1.0")
"""

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Migration function type: takes config dict, returns new config dict
MigrationFunc = Callable[[Dict[str, Any]], Dict[str, Any]]


@dataclass
class MigrationRule:
    """A single migration rule between versions."""
    from_version: str
    to_version: str
    description: str
    migrate_fn: MigrationFunc
    created_at: float = 0.0

    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()


@dataclass
class MigrationResult:
    """Result of applying migrations."""
    original_version: str
    final_version: str
    applied_migrations: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    success: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_version": self.original_version,
            "final_version": self.final_version,
            "applied_migrations": self.applied_migrations,
            "warnings": self.warnings,
            "success": self.success,
        }


# ── Built-in migrations ──────────────────────────────

def _migrate_v1_to_v2(config: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate from v1 (flat .env) to v2 (structured config)."""
    new = dict(config)

    # Rename old keys
    renames = {
        "OPENAI_KEY": "OPENAI_API_KEY",
        "ANTHROPIC_KEY": "ANTHROPIC_API_KEY",
        "BRAVE_KEY": "BRAVE_API_KEY",
        "TG_TOKEN": "TELEGRAM_BOT_TOKEN",
        "TG_BOT_TOKEN": "TELEGRAM_BOT_TOKEN",
        "DB_URL": "DATABASE_URL",
        "DB_HOST": None,  # Removed; use DATABASE_URL
        "DB_PORT": None,
        "DB_NAME": None,
        "DB_USER": None,
        "DB_PASS": None,
        "MODEL": "AGENT_MODEL",
        "FALLBACK_MODEL": "AGENT_FALLBACK_MODEL",
        "MAX_TOKENS": "AGENT_MAX_TOKENS",
        "TOOL_ITERATIONS": "AGENT_MAX_TOOL_ITERATIONS",
    }

    for old_key, new_key in renames.items():
        if old_key in new:
            value = new.pop(old_key)
            if new_key and new_key not in new:
                new[new_key] = value

    return new


def _migrate_v2_to_v3(config: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate from v2 to v3 — add new defaults, restructure channels."""
    new = dict(config)

    # Ensure new required fields have defaults
    defaults = {
        "THINKING_BUDGET_DEFAULT": "4096",
        "TTS_AUTO_MODE": "off",
        "DM_POLICY": "open",
        "GROUP_POLICY": "open",
        "SANDBOX_ENABLED": "false",
        "HEARTBEAT_ENABLED": "true",
    }

    for key, default_val in defaults.items():
        if key not in new:
            new[key] = default_val

    return new


class ConfigMigrationManager:
    """
    Manages config version migrations.

    Migrations are registered as (from_version, to_version) pairs.
    When migrate() is called, it finds the path from the current
    version to the latest and applies all intermediate migrations.
    """

    CURRENT_VERSION = "3.0"

    def __init__(self):
        self._migrations: List[MigrationRule] = []
        self._register_builtins()

    def _register_builtins(self):
        """Register built-in migration rules."""
        self.register_migration(
            "1.0", "2.0",
            _migrate_v1_to_v2,
            "Rename legacy env vars to standardized names",
        )
        self.register_migration(
            "2.0", "3.0",
            _migrate_v2_to_v3,
            "Add new v3 defaults (thinking, TTS, policies, sandbox)",
        )

    def register_migration(
        self,
        from_version: str,
        to_version: str,
        migrate_fn: MigrationFunc,
        description: str = "",
    ) -> MigrationRule:
        """Register a migration rule."""
        rule = MigrationRule(
            from_version=from_version,
            to_version=to_version,
            description=description,
            migrate_fn=migrate_fn,
        )
        self._migrations.append(rule)
        logger.info(f"[MIGRATION] Registered: {from_version} → {to_version}: {description}")
        return rule

    def get_migration_path(self, from_version: str, to_version: Optional[str] = None) -> List[MigrationRule]:
        """Find the ordered list of migrations to apply."""
        target = to_version or self.CURRENT_VERSION
        path = []
        current = from_version

        seen = set()
        while current != target:
            if current in seen:
                break  # Circular dependency guard
            seen.add(current)

            found = False
            for rule in self._migrations:
                if rule.from_version == current:
                    path.append(rule)
                    current = rule.to_version
                    found = True
                    break

            if not found:
                break

        return path

    def migrate(
        self,
        config: Dict[str, Any],
        from_version: str = "1.0",
        to_version: Optional[str] = None,
    ) -> MigrationResult:
        """
        Apply all necessary migrations to a config dict.

        Args:
            config: The current configuration dict.
            from_version: The current config version.
            to_version: Target version (defaults to CURRENT_VERSION).

        Returns:
            MigrationResult with the migrated config.
        """
        target = to_version or self.CURRENT_VERSION
        result = MigrationResult(
            original_version=from_version,
            final_version=from_version,
            config=copy.deepcopy(config),
        )

        path = self.get_migration_path(from_version, target)
        if not path:
            result.final_version = from_version
            if from_version != target:
                result.warnings.append(f"No migration path from {from_version} to {target}")
            return result

        for rule in path:
            try:
                result.config = rule.migrate_fn(result.config)
                result.applied_migrations.append(f"{rule.from_version} → {rule.to_version}")
                result.final_version = rule.to_version
                logger.info(f"[MIGRATION] Applied: {rule.from_version} → {rule.to_version}")
            except Exception as e:
                result.success = False
                result.warnings.append(f"Migration {rule.from_version}→{rule.to_version} failed: {e}")
                logger.error(f"[MIGRATION] Failed: {rule.from_version} → {rule.to_version}: {e}")
                break

        return result

    def detect_version(self, config: Dict[str, Any]) -> str:
        """Auto-detect config version from key patterns."""
        # v1 keys (old names)
        v1_keys = {"OPENAI_KEY", "TG_TOKEN", "DB_URL", "MODEL"}
        if any(k in config for k in v1_keys):
            return "1.0"

        # v3 keys (new features)
        v3_keys = {"THINKING_BUDGET_DEFAULT", "DM_POLICY", "SANDBOX_ENABLED"}
        if any(k in config for k in v3_keys):
            return "3.0"

        # Default to v2
        return "2.0"

    def list_migrations(self) -> List[Dict[str, str]]:
        """List all registered migration rules."""
        return [
            {
                "from": r.from_version,
                "to": r.to_version,
                "description": r.description,
            }
            for r in self._migrations
        ]

    @property
    def migration_count(self) -> int:
        return len(self._migrations)


# ── Singleton ────────────────────────────────────────────
_manager: Optional[ConfigMigrationManager] = None


def get_migration_manager() -> ConfigMigrationManager:
    """Get the global config migration manager."""
    global _manager
    if _manager is None:
        _manager = ConfigMigrationManager()
    return _manager
