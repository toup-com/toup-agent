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
    # Dynamic registration (hot-reload)
    # ------------------------------------------------------------------

    async def register_dynamic(self, skill: Skill) -> bool:
        """Register a skill instance at runtime (no filesystem needed).
        If a skill with the same name already exists, unload it first.

        Returns False when the skill belongs to a tool family this tenant is
        not entitled to (see `_register`)."""
        name = skill.meta.name
        if name in self._skills:
            await self.unload_skill(name)
        return await self._register(skill)

    async def unload_skill(self, name: str) -> bool:
        """Unload a single skill by name."""
        skill = self._skills.pop(name, None)
        if not skill:
            return False
        # Remove tool index entries
        for tool in skill.get_tools():
            self._tool_index.pop(tool.get("name", ""), None)
        try:
            await skill.on_unload()
        except Exception as e:
            logger.warning(f"[SKILLS] Error unloading {name}: {e}")
        logger.info(f"[SKILLS] Unloaded '{name}'")
        return True

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

    async def on_tool_input(
        self,
        tool_name: str,
        call_id: str,
        partial_json: str,
        ctx: SkillContext,
    ) -> None:
        """Route still-arriving tool arguments to the owning skill.

        Round 25 — see :meth:`Skill.on_tool_input`. Deliberately silent on
        every failure INCLUDING an unknown tool: this runs inside the token
        loop, on a stream that is still open, for the sole purpose of making a
        progress card appear sooner. Nothing it can discover is worth an
        exception reaching the runner, and a skill that does not implement the
        hook inherits the base no-op.
        """
        skill_name = self._tool_index.get(tool_name)
        if not skill_name:
            return
        skill = self._skills.get(skill_name)
        if not skill:
            return
        try:
            await skill.on_tool_input(tool_name, call_id, partial_json, ctx)
        except Exception:  # noqa: BLE001 - a progress hint never kills a turn
            logger.debug("[SKILLS] on_tool_input failed for %s", tool_name,
                         exc_info=True)

    # ------------------------------------------------------------------
    # Discovery & loading
    # ------------------------------------------------------------------

    async def load_all(self) -> int:
        """Discover and load skills from builtin + external dirs. Returns count loaded."""
        # 1. Builtins (siblings of this file under builtins/)
        builtins_dir = os.path.join(os.path.dirname(__file__), "builtins")
        dirs_to_scan = [builtins_dir] + self._extra_dirs

        # Imported here rather than at module scope for the same reason
        # `_register` imports `skill_enabled` locally: `tool_entitlements`
        # reaches into `app.config`, and the loader is constructed early
        # enough in boot that a top-level import would order-couple the two.
        from app.agent.tool_entitlements import RETIRED_SKILLS

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

                # A retired skill is skipped BEFORE the import, not after.
                # `_register` would reject it anyway, but `_load_skill_from_file`
                # execs the module to find the Skill subclass, and
                # `app_builder/skill.py` is 278 KB whose import pulls in the
                # AppManager/Metro machinery and leaves a `toup_skill_*` entry
                # in `sys.modules`. Nothing should pay that to load a skill
                # that cannot register.
                if entry in RETIRED_SKILLS:
                    logger.info(
                        "[SKILLS] Skipping retired skill dir '%s' — not imported",
                        entry,
                    )
                    continue

                try:
                    skill = self._load_skill_from_file(skill_file, entry)
                    if skill and await self._register(skill):
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
        module_name = f"toup_skill_{fallback_name}"

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

    async def _register(self, skill: Skill) -> bool:
        """Validate and register a skill + index its tools.

        Returns True when the skill ended up registered.

        This is the single funnel for BOTH `load_all()` (filesystem
        discovery) and `register_dynamic()` (agent_main's late-bound
        AppBuilderSkill), which is why the per-tenant entitlement check sits
        here: withholding a skill at registration removes its tool
        definitions, its system-prompt section, its commands, its hooks AND
        its execution path together. A skill that is half-gated — tools
        hidden but still callable, or callable but absent from the
        capabilities listing — is worse than either extreme.

        The entitlement is resolved once per process, so the set of loaded
        skills is fixed for the life of the container. That matters because
        skill tools are appended to the wire tools array, which heads the
        provider cache prefix: a skill that came and went mid-life would
        fork the cache lineage ([PERF] tools_array_changed).
        """
        name = skill.meta.name

        if name in self._skills:
            logger.warning(f"[SKILLS] Duplicate skill name '{name}' — skipping")
            return False

        from app.agent.tool_entitlements import skill_enabled

        if not skill_enabled(name):
            logger.info(
                "[SKILLS] Skipping '%s' — its tool family is not entitled for "
                "this tenant (AGENT_TOOL_FAMILIES)", name,
            )
            return False

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
        return True
