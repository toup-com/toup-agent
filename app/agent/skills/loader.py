"""
SkillLoader — discovers, validates, and manages skill plugins.

Skills are loaded from:
  1. Built-in directory: backend/app/agent/skills/builtins/
  2. External directory: settings.skills_dir (default /app/skills)

Each skill directory must contain a `skill.py` with a class that extends `Skill`.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.agent.skills.base import Skill, SkillContext, SkillMeta

logger = logging.getLogger(__name__)


class SkillLoader:
    """Discovers, loads, and manages skill plugins."""

    def __init__(self, extra_dirs: Optional[List[str]] = None):
        self._skills: Dict[str, Skill] = {}  # keyed by skill.meta.name
        self._tool_index: Dict[str, str] = {}  # tool_name → skill_name
        self._extra_dirs = extra_dirs or []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def skills(self) -> Dict[str, Skill]:
        return dict(self._skills)

    @property
    def loaded_count(self) -> int:
        return len(self._skills)

    def get_skill(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def is_skill_tool(self, tool_name: str) -> bool:
        """Check if a tool name belongs to a skill (contains __ prefix)."""
        return tool_name in self._tool_index

    def get_all_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return combined tool definitions from all loaded skills."""
        tools: List[Dict[str, Any]] = []
        for skill in self._skills.values():
            tools.extend(skill.get_tools())
        return tools

    def get_all_system_prompt_sections(self) -> List[str]:
        """Collect system prompt sections from all loaded skills."""
        sections: List[str] = []
        for skill in self._skills.values():
            section = skill.get_system_prompt_section()
            if section:
                sections.append(section)
        return sections

    def get_summary(self) -> List[Dict[str, Any]]:
        """Return a summary of all loaded skills for /skills command."""
        out: List[Dict[str, Any]] = []
        for skill in self._skills.values():
            tool_names = [t["name"] for t in skill.get_tools()]
            out.append({
                "name": skill.meta.name,
                "version": skill.meta.version,
                "description": skill.meta.description,
                "author": skill.meta.author,
                "tools": tool_names,
            })
        return out

    def get_all_commands(self) -> List[Dict[str, Any]]:
        """Collect chat commands from all loaded skills."""
        commands: List[Dict[str, Any]] = []
        for skill in self._skills.values():
            try:
                cmds = skill.get_commands()
                for cmd in cmds:
                    cmd["skill"] = skill.meta.name
                commands.extend(cmds)
            except Exception:
                pass
        return commands

    def get_all_hooks(self) -> List[Dict[str, Any]]:
        """Collect lifecycle hooks from all loaded skills."""
        hooks: List[Dict[str, Any]] = []
        for skill in self._skills.values():
            try:
                hks = skill.get_hooks()
                for hk in hks:
                    hk["skill"] = skill.meta.name
                hooks.extend(hks)
            except Exception:
                pass
        return hooks

    # ------------------------------------------------------------------
    # Tool execution
    # ------------------------------------------------------------------

    async def execute_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
        ctx: SkillContext,
    ) -> str:
        """Route a tool call to the owning skill."""
        skill_name = self._tool_index.get(tool_name)
        if not skill_name:
            return f"ERROR: Unknown skill tool: {tool_name}"

        skill = self._skills.get(skill_name)
        if not skill:
            return f"ERROR: Skill '{skill_name}' not loaded"

        try:
            return await skill.execute_tool(tool_name, args, ctx)
        except Exception as e:
            logger.exception(f"Skill tool {tool_name} crashed")
            return f"ERROR: Skill tool crashed: {type(e).__name__}: {e}"

    # ------------------------------------------------------------------
    # Discovery & loading
    # ------------------------------------------------------------------

    async def load_all(self) -> int:
        """Discover and load skills from builtin + external dirs. Returns count loaded."""
        # 1. Builtins (siblings of this file under builtins/)
        builtins_dir = os.path.join(os.path.dirname(__file__), "builtins")
        dirs_to_scan = [builtins_dir] + self._extra_dirs

        loaded = 0
        for scan_dir in dirs_to_scan:
            if not os.path.isdir(scan_dir):
                logger.debug(f"[SKILLS] Skipping non-existent dir: {scan_dir}")
                continue

            for entry in sorted(os.listdir(scan_dir)):
                skill_dir = os.path.join(scan_dir, entry)
                skill_file = os.path.join(skill_dir, "skill.py")

                if not os.path.isfile(skill_file):
                    continue

                try:
                    skill = self._load_skill_from_file(skill_file, entry)
                    if skill:
                        await self._register(skill)
                        loaded += 1
                except Exception as e:
                    logger.error(f"[SKILLS] Failed to load skill from {skill_dir}: {e}")

        logger.info(f"[SKILLS] Loaded {loaded} skill(s): {list(self._skills.keys())}")
        return loaded

    async def unload_all(self) -> None:
        """Unload all skills (calls on_unload hooks)."""
        for name, skill in list(self._skills.items()):
            try:
                await skill.on_unload()
            except Exception as e:
                logger.warning(f"[SKILLS] Error unloading {name}: {e}")
        self._skills.clear()
        self._tool_index.clear()
        logger.info("[SKILLS] All skills unloaded")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_skill_from_file(self, filepath: str, fallback_name: str) -> Optional[Skill]:
        """Import a skill.py file and find the Skill subclass."""
        module_name = f"hexbrain_skill_{fallback_name}"

        spec = importlib.util.spec_from_file_location(module_name, filepath)
        if not spec or not spec.loader:
            logger.warning(f"[SKILLS] Cannot create module spec for {filepath}")
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)
        except Exception as e:
            logger.error(f"[SKILLS] Error importing {filepath}: {e}")
            del sys.modules[module_name]
            return None

        # Find the Skill subclass in the module
        skill_class = None
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Skill) and obj is not Skill:
                skill_class = obj
                break

        if not skill_class:
            logger.warning(f"[SKILLS] No Skill subclass found in {filepath}")
            return None

        try:
            instance = skill_class()
        except Exception as e:
            logger.error(f"[SKILLS] Error instantiating {skill_class.__name__}: {e}")
            return None

        if not hasattr(instance, "meta") or not isinstance(instance.meta, SkillMeta):
            logger.warning(f"[SKILLS] {skill_class.__name__} missing valid 'meta' attribute")
            return None

        return instance

    async def _register(self, skill: Skill) -> None:
        """Validate and register a skill + index its tools."""
        name = skill.meta.name

        if name in self._skills:
            logger.warning(f"[SKILLS] Duplicate skill name '{name}' — skipping")
            return

        tools = skill.get_tools()
        prefix = f"{name}__"

        for tool in tools:
            tool_name = tool.get("name", "")
            if not tool_name.startswith(prefix):
                raise ValueError(
                    f"Skill '{name}' tool '{tool_name}' must start with '{prefix}'"
                )
            if tool_name in self._tool_index:
                raise ValueError(
                    f"Tool name collision: '{tool_name}' already registered by "
                    f"skill '{self._tool_index[tool_name]}'"
                )
            self._tool_index[tool_name] = name

        # Call lifecycle hook
        await skill.on_load()

        self._skills[name] = skill
        logger.info(
            f"[SKILLS] Registered '{name}' v{skill.meta.version} "
            f"({len(tools)} tools: {[t['name'] for t in tools]})"
        )
