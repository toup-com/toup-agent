"""
Skill Marketplace — Remote skill registry with install/search/update.

Provides:
  * Search for skills from a remote registry
  * Install / uninstall skills
  * Update installed skills
  * Skill manifest with dependencies
  * Compatibility checking
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class SkillStatus(str, Enum):
    AVAILABLE = "available"
    INSTALLED = "installed"
    UPDATE_AVAILABLE = "update_available"
    INCOMPATIBLE = "incompatible"


@dataclass
class SkillManifest:
    """Manifest for a marketplace skill."""
    name: str
    version: str
    description: str
    author: str = ""
    homepage: str = ""
    license: str = "MIT"
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)  # pip packages
    min_platform_version: str = "1.0.0"
    entry_point: str = "__init__.py"
    tools: List[str] = field(default_factory=list)  # Tool names provided
    commands: List[str] = field(default_factory=list)  # Commands provided
    icon: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "homepage": self.homepage,
            "license": self.license,
            "tags": self.tags,
            "dependencies": self.dependencies,
            "min_platform_version": self.min_platform_version,
            "entry_point": self.entry_point,
            "tools": self.tools,
            "commands": self.commands,
            "icon": self.icon,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillManifest":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class InstalledSkill:
    """Tracking info for an installed skill."""
    manifest: SkillManifest
    installed_at: float = field(default_factory=time.time)
    install_path: str = ""
    enabled: bool = True


class SkillMarketplace:
    """
    Manages skill discovery, installation, and updates.
    """

    PLATFORM_VERSION = "1.6.0"

    def __init__(self, skills_dir: str = "/app/skills",
                 registry_url: str = ""):
        self.skills_dir = skills_dir
        self.registry_url = registry_url
        self._installed: Dict[str, InstalledSkill] = {}
        self._registry_cache: List[SkillManifest] = []
        self._cache_time: float = 0
        self._cache_ttl: float = 300  # 5 minute cache

        # Scan existing installed skills
        self._scan_installed()

    def _scan_installed(self):
        """Scan the skills directory for installed skills."""
        if not os.path.isdir(self.skills_dir):
            return

        for name in os.listdir(self.skills_dir):
            skill_dir = os.path.join(self.skills_dir, name)
            manifest_path = os.path.join(skill_dir, "manifest.json")
            if os.path.isfile(manifest_path):
                try:
                    with open(manifest_path) as f:
                        data = json.load(f)
                    manifest = SkillManifest.from_dict(data)
                    self._installed[name] = InstalledSkill(
                        manifest=manifest,
                        install_path=skill_dir,
                    )
                except Exception as e:
                    logger.warning("[MARKETPLACE] Failed to load manifest for %s: %s", name, e)

    async def search(self, query: str = "", tags: Optional[List[str]] = None,
                     limit: int = 20) -> List[Dict[str, Any]]:
        """Search the remote registry for skills."""
        registry = await self._fetch_registry()

        results = []
        for manifest in registry:
            # Filter by query
            if query:
                q = query.lower()
                searchable = f"{manifest.name} {manifest.description} {' '.join(manifest.tags)}".lower()
                if q not in searchable:
                    continue

            # Filter by tags
            if tags:
                if not any(t in manifest.tags for t in tags):
                    continue

            # Determine status
            status = SkillStatus.AVAILABLE
            if manifest.name in self._installed:
                installed = self._installed[manifest.name]
                if installed.manifest.version < manifest.version:
                    status = SkillStatus.UPDATE_AVAILABLE
                else:
                    status = SkillStatus.INSTALLED

            if not self._check_compatibility(manifest):
                status = SkillStatus.INCOMPATIBLE

            results.append({
                **manifest.to_dict(),
                "status": status.value,
            })

            if len(results) >= limit:
                break

        return results

    async def install(self, skill_name: str) -> Dict[str, str]:
        """Install a skill from the registry."""
        if skill_name in self._installed:
            return {"error": f"Skill '{skill_name}' is already installed"}

        registry = await self._fetch_registry()
        manifest = next((m for m in registry if m.name == skill_name), None)

        if not manifest:
            return {"error": f"Skill '{skill_name}' not found in registry"}

        if not self._check_compatibility(manifest):
            return {"error": f"Skill '{skill_name}' requires platform >= {manifest.min_platform_version}"}

        # Create skill directory
        skill_dir = os.path.join(self.skills_dir, skill_name)
        os.makedirs(skill_dir, exist_ok=True)

        # Download skill files
        try:
            files = await self._download_skill(skill_name, manifest)
            for filename, content in files.items():
                filepath = os.path.join(skill_dir, filename)
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                with open(filepath, "w") as f:
                    f.write(content)

            # Save manifest
            with open(os.path.join(skill_dir, "manifest.json"), "w") as f:
                json.dump(manifest.to_dict(), f, indent=2)

            self._installed[skill_name] = InstalledSkill(
                manifest=manifest,
                install_path=skill_dir,
            )

            logger.info("[MARKETPLACE] Installed skill: %s v%s", skill_name, manifest.version)
            return {
                "status": "installed",
                "name": skill_name,
                "version": manifest.version,
                "tools": manifest.tools,
            }
        except Exception as e:
            # Cleanup on failure
            if os.path.isdir(skill_dir):
                shutil.rmtree(skill_dir, ignore_errors=True)
            logger.error("[MARKETPLACE] Install failed for %s: %s", skill_name, e)
            return {"error": f"Install failed: {e}"}

    async def uninstall(self, skill_name: str) -> Dict[str, str]:
        """Uninstall a skill."""
        installed = self._installed.pop(skill_name, None)
        if not installed:
            return {"error": f"Skill '{skill_name}' is not installed"}

        if installed.install_path and os.path.isdir(installed.install_path):
            shutil.rmtree(installed.install_path, ignore_errors=True)

        logger.info("[MARKETPLACE] Uninstalled skill: %s", skill_name)
        return {"status": "uninstalled", "name": skill_name}

    async def update(self, skill_name: str) -> Dict[str, str]:
        """Update an installed skill to the latest version."""
        if skill_name not in self._installed:
            return {"error": f"Skill '{skill_name}' is not installed"}

        # Uninstall then reinstall
        await self.uninstall(skill_name)
        return await self.install(skill_name)

    def list_installed(self) -> List[Dict[str, Any]]:
        """List all installed skills."""
        return [
            {
                **info.manifest.to_dict(),
                "status": "installed",
                "installed_at": info.installed_at,
                "enabled": info.enabled,
                "install_path": info.install_path,
            }
            for name, info in self._installed.items()
        ]

    def get_installed(self, skill_name: str) -> Optional[InstalledSkill]:
        """Get info about an installed skill."""
        return self._installed.get(skill_name)

    def enable_skill(self, skill_name: str) -> bool:
        """Enable an installed skill."""
        info = self._installed.get(skill_name)
        if info:
            info.enabled = True
            return True
        return False

    def disable_skill(self, skill_name: str) -> bool:
        """Disable an installed skill."""
        info = self._installed.get(skill_name)
        if info:
            info.enabled = False
            return True
        return False

    def _check_compatibility(self, manifest: SkillManifest) -> bool:
        """Check if a skill is compatible with this platform version."""
        try:
            req = tuple(int(x) for x in manifest.min_platform_version.split("."))
            cur = tuple(int(x) for x in self.PLATFORM_VERSION.split("."))
            return cur >= req
        except ValueError:
            return True

    async def _fetch_registry(self) -> List[SkillManifest]:
        """Fetch the skill registry (with caching)."""
        now = time.time()
        if self._registry_cache and (now - self._cache_time) < self._cache_ttl:
            return self._registry_cache

        if not self.registry_url:
            # Return built-in demo registry
            return self._builtin_registry()

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(self.registry_url)
                resp.raise_for_status()
                data = resp.json()

            self._registry_cache = [
                SkillManifest.from_dict(item) for item in data.get("skills", [])
            ]
            self._cache_time = now
            return self._registry_cache
        except Exception as e:
            logger.warning("[MARKETPLACE] Failed to fetch registry: %s", e)
            return self._builtin_registry()

    def _builtin_registry(self) -> List[SkillManifest]:
        """Built-in skill registry for demo / offline mode."""
        return [
            SkillManifest(
                name="math_tools",
                version="1.0.0",
                description="Advanced mathematical tools: symbolic math, plotting, statistics",
                author="HexBrain",
                tags=["math", "science", "plotting"],
                tools=["math_eval", "math_plot", "math_stats"],
            ),
            SkillManifest(
                name="github_integration",
                version="1.0.0",
                description="GitHub API integration: repos, issues, PRs, actions",
                author="HexBrain",
                tags=["github", "git", "devops"],
                dependencies=["PyGithub"],
                tools=["github_repos", "github_issues", "github_pr"],
            ),
            SkillManifest(
                name="database_tools",
                version="1.0.0",
                description="Database management: SQL execution, schema inspection, migrations",
                author="HexBrain",
                tags=["database", "sql", "postgres"],
                tools=["db_query", "db_schema", "db_migrate"],
            ),
            SkillManifest(
                name="image_generation",
                version="1.0.0",
                description="AI image generation via DALL-E, Stable Diffusion",
                author="HexBrain",
                tags=["image", "ai", "generation"],
                dependencies=["openai"],
                tools=["img_generate", "img_edit", "img_variation"],
            ),
            SkillManifest(
                name="calendar_sync",
                version="1.0.0",
                description="Calendar integration: Google Calendar, Outlook events",
                author="HexBrain",
                tags=["calendar", "schedule", "productivity"],
                tools=["cal_events", "cal_create", "cal_update"],
            ),
        ]

    async def _download_skill(self, skill_name: str,
                              manifest: SkillManifest) -> Dict[str, str]:
        """Download skill files from the registry."""
        if not self.registry_url:
            # Demo mode: generate a stub skill
            return {
                "__init__.py": f'''"""Skill: {manifest.name} v{manifest.version}"""
from app.agent.skills.base import Skill, SkillMeta

class {skill_name.title().replace("_", "")}Skill(Skill):
    @property
    def meta(self) -> SkillMeta:
        return SkillMeta(
            name="{manifest.name}",
            version="{manifest.version}",
            description="{manifest.description}",
            author="{manifest.author}",
        )

    def get_tools(self):
        return {json.dumps([{"name": t, "description": f"{t} tool", "input_schema": {"type": "object", "properties": {}}} for t in manifest.tools])}

    async def execute_tool(self, tool_name, tool_input, context):
        return f"{{tool_name}} executed (stub)"
''',
                "manifest.json": json.dumps(manifest.to_dict(), indent=2),
            }

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(f"{self.registry_url}/skills/{skill_name}/files")
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            raise RuntimeError(f"Download failed: {e}")


# Global singleton
_marketplace: Optional[SkillMarketplace] = None


def get_marketplace(skills_dir: str = "/app/skills") -> SkillMarketplace:
    global _marketplace
    if _marketplace is None:
        _marketplace = SkillMarketplace(skills_dir=skills_dir)
    return _marketplace
