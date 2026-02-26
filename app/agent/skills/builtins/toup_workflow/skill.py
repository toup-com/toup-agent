"""
Toup Workflow Skill — Knowledge and guidance for building HexBrain visual workflows.

Provides the agent with deep knowledge about the 25 node types, workflow patterns,
validation rules, MCP tool usage, and common mistakes — modeled after the
n8n-skills architecture (SKILL.md + reference docs).

Tools:
  toup_workflow__get_skill_guide       — Core workflow-building knowledge
  toup_workflow__get_node_reference    — Complete node catalog (25 nodes)
  toup_workflow__get_pattern           — Proven workflow patterns with examples
  toup_workflow__get_validation_guide  — Validation rules & error fixing
  toup_workflow__get_tools_guide       — MCP tool reference (16 tools)
  toup_workflow__get_common_mistakes   — Error catalog with solutions
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.agent.skills.base import Skill, SkillContext, SkillMeta

logger = logging.getLogger(__name__)

# Directory containing this skill's .md knowledge files
SKILL_DIR = Path(__file__).parent


def _load_md(filename: str) -> str:
    """Load a markdown file from the skill directory."""
    path = SKILL_DIR / filename
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("Skill knowledge file not found: %s", path)
        return f"ERROR: Knowledge file '{filename}' not found."


# Pre-load at import time so files are cached
_KNOWLEDGE: Dict[str, str] = {}


def _get_knowledge(key: str) -> str:
    """Lazy-load and cache knowledge files."""
    if key not in _KNOWLEDGE:
        _KNOWLEDGE[key] = _load_md(key)
    return _KNOWLEDGE[key]


# Map of topic keywords → knowledge file
_TOPIC_MAP = {
    "skill": "SKILL.md",
    "guide": "SKILL.md",
    "overview": "SKILL.md",
    "node": "NODE_REFERENCE.md",
    "nodes": "NODE_REFERENCE.md",
    "reference": "NODE_REFERENCE.md",
    "template": "NODE_REFERENCE.md",
    "pattern": "WORKFLOW_PATTERNS.md",
    "patterns": "WORKFLOW_PATTERNS.md",
    "example": "WORKFLOW_PATTERNS.md",
    "chatbot": "WORKFLOW_PATTERNS.md",
    "webhook": "WORKFLOW_PATTERNS.md",
    "validation": "VALIDATION_GUIDE.md",
    "validate": "VALIDATION_GUIDE.md",
    "error": "VALIDATION_GUIDE.md",
    "errors": "VALIDATION_GUIDE.md",
    "tool": "MCP_TOOLS_GUIDE.md",
    "tools": "MCP_TOOLS_GUIDE.md",
    "mcp": "MCP_TOOLS_GUIDE.md",
    "mistake": "COMMON_MISTAKES.md",
    "mistakes": "COMMON_MISTAKES.md",
    "common": "COMMON_MISTAKES.md",
    "fix": "COMMON_MISTAKES.md",
    "handle": "COMMON_MISTAKES.md",
}


class ToupWorkflowSkill(Skill):
    """Workflow building knowledge — nodes, patterns, validation, tools, mistakes."""

    meta = SkillMeta(
        name="toup_workflow",
        version="1.0.0",
        description=(
            "Deep knowledge for building HexBrain visual workflows: "
            "25 node types, 6 patterns, validation rules, MCP tool guidance, "
            "and common mistake avoidance."
        ),
        author="Toup",
    )

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------
    def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "toup_workflow__get_skill_guide",
                "description": (
                    "Get the core workflow-building guide: node quick reference, "
                    "building sequence, core concepts (triggers, handles, config, "
                    "status lifecycle), common patterns overview, canvas layout tips."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "toup_workflow__get_node_reference",
                "description": (
                    "Get the complete reference for all 25 workflow node types. "
                    "Includes fields, handles, defaults, use cases for every node "
                    "across 7 categories: triggers, AI, actions, logic, data, output."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": (
                                "Optional: filter by category. "
                                "One of: triggers, ai, actions, logic, data, output. "
                                "Omit to get all nodes."
                            ),
                            "enum": ["triggers", "ai", "actions", "logic", "data", "output"],
                        },
                    },
                    "required": [],
                },
            },
            {
                "name": "toup_workflow__get_pattern",
                "description": (
                    "Get proven workflow patterns with complete step-by-step "
                    "build instructions. Patterns: chatbot, webhook_pipeline, "
                    "smart_router, scheduled_task, loop_processing, fan_out_fan_in."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "pattern_name": {
                            "type": "string",
                            "description": (
                                "Optional: specific pattern to retrieve. "
                                "One of: chatbot, webhook_pipeline, smart_router, "
                                "scheduled_task, loop_processing, fan_out_fan_in. "
                                "Omit to get all patterns."
                            ),
                            "enum": [
                                "chatbot",
                                "webhook_pipeline",
                                "smart_router",
                                "scheduled_task",
                                "loop_processing",
                                "fan_out_fan_in",
                            ],
                        },
                    },
                    "required": [],
                },
            },
            {
                "name": "toup_workflow__get_validation_guide",
                "description": (
                    "Get the workflow validation guide: all validation rules, "
                    "required fields per node, valid handle names, error categories, "
                    "fixing strategies, and pre-activation checklist."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "toup_workflow__get_tools_guide",
                "description": (
                    "Get the MCP tools guide: all 16 workflow tools with parameters, "
                    "return values, examples, 2 resources, 3 prompts, and "
                    "common tool sequences for build/debug/clone workflows."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "toup_workflow__get_common_mistakes",
                "description": (
                    "Get the common mistakes catalog: connection mistakes (handles), "
                    "config mistakes, structure mistakes, ordering mistakes, "
                    "and canvas layout mistakes — each with the wrong/right pattern."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
            {
                "name": "toup_workflow__lookup",
                "description": (
                    "Smart lookup across all workflow knowledge. Provide a topic "
                    "keyword and get the most relevant guide section. "
                    "Keywords: node, pattern, validation, tool, mistake, handle, "
                    "chatbot, webhook, error, fix, mcp, template, guide."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "topic": {
                            "type": "string",
                            "description": (
                                "Topic keyword to look up. Examples: 'node', "
                                "'pattern', 'validation', 'handle', 'chatbot', "
                                "'error', 'mcp', 'fix'."
                            ),
                        },
                    },
                    "required": ["topic"],
                },
            },
        ]

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------
    def get_system_prompt_section(self) -> Optional[str]:
        return (
            "# Toup Workflow Skill — Visual Workflow Builder Knowledge\n\n"
            "You have deep knowledge about building HexBrain visual workflows.\n"
            "Use the `toup_workflow__*` tools to access detailed guides:\n\n"
            "- `toup_workflow__get_skill_guide` — Core concepts, quick reference, building sequence\n"
            "- `toup_workflow__get_node_reference` — All 25 node types with fields & handles\n"
            "- `toup_workflow__get_pattern` — 6 proven patterns with step-by-step examples\n"
            "- `toup_workflow__get_validation_guide` — Validation rules, errors, fixes\n"
            "- `toup_workflow__get_tools_guide` — All 16 MCP tools with usage patterns\n"
            "- `toup_workflow__get_common_mistakes` — Error catalog with wrong/right patterns\n"
            "- `toup_workflow__lookup` — Smart keyword search across all knowledge\n\n"
            "## Quick Rules\n"
            "1. Every workflow needs exactly ONE trigger node\n"
            "2. IF node → specify `source_handle`: 'true' or 'false'\n"
            "3. Switch node → specify `source_handle`: 'case1', 'case2', 'case3', 'default'\n"
            "4. Loop node → specify `source_handle`: 'item' or 'done'\n"
            "5. Merge node → specify `target_handle`: 'in1' or 'in2'\n"
            "6. Always validate before activating: validate_workflow → set_workflow_status\n"
            "7. Space nodes ~300px apart horizontally on the canvas\n\n"
            "When building workflows, ALWAYS consult these guides to ensure correct "
            "node configuration, handle routing, and validation."
        )

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------
    async def execute_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
        ctx: SkillContext,
    ) -> str:
        dispatch = {
            "toup_workflow__get_skill_guide": self._get_skill_guide,
            "toup_workflow__get_node_reference": self._get_node_reference,
            "toup_workflow__get_pattern": self._get_pattern,
            "toup_workflow__get_validation_guide": self._get_validation_guide,
            "toup_workflow__get_tools_guide": self._get_tools_guide,
            "toup_workflow__get_common_mistakes": self._get_common_mistakes,
            "toup_workflow__lookup": self._lookup,
        }
        handler = dispatch.get(tool_name)
        if not handler:
            return f"ERROR: Unknown toup_workflow tool: {tool_name}"
        return await handler(args, ctx)

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    async def _get_skill_guide(
        self, args: Dict[str, Any], ctx: SkillContext
    ) -> str:
        return _get_knowledge("SKILL.md")

    async def _get_node_reference(
        self, args: Dict[str, Any], ctx: SkillContext
    ) -> str:
        content = _get_knowledge("NODE_REFERENCE.md")
        category = args.get("category")
        if category:
            # Extract just the relevant section
            return self._extract_section(content, category)
        return content

    async def _get_pattern(
        self, args: Dict[str, Any], ctx: SkillContext
    ) -> str:
        content = _get_knowledge("WORKFLOW_PATTERNS.md")
        pattern_name = args.get("pattern_name")
        if pattern_name:
            return self._extract_pattern(content, pattern_name)
        return content

    async def _get_validation_guide(
        self, args: Dict[str, Any], ctx: SkillContext
    ) -> str:
        return _get_knowledge("VALIDATION_GUIDE.md")

    async def _get_tools_guide(
        self, args: Dict[str, Any], ctx: SkillContext
    ) -> str:
        return _get_knowledge("MCP_TOOLS_GUIDE.md")

    async def _get_common_mistakes(
        self, args: Dict[str, Any], ctx: SkillContext
    ) -> str:
        return _get_knowledge("COMMON_MISTAKES.md")

    async def _lookup(
        self, args: Dict[str, Any], ctx: SkillContext
    ) -> str:
        topic = args.get("topic", "").lower().strip()
        if not topic:
            return "ERROR: 'topic' is required. Try: node, pattern, validation, tool, mistake, handle"

        # Try direct match
        filename = _TOPIC_MAP.get(topic)
        if filename:
            return _get_knowledge(filename)

        # Try substring match
        for keyword, fname in _TOPIC_MAP.items():
            if keyword in topic or topic in keyword:
                return _get_knowledge(fname)

        # Fallback: return the main skill guide
        return (
            f"No specific guide found for '{topic}'. "
            "Here's the main skill guide:\n\n"
            + _get_knowledge("SKILL.md")
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_section(content: str, category: str) -> str:
        """Extract a category section from NODE_REFERENCE.md."""
        # Map category to section headers in the doc
        header_map = {
            "triggers": "Trigger Nodes",
            "ai": "AI Nodes",
            "actions": "Action Nodes",
            "logic": "Logic Nodes",
            "data": "Data Nodes",
            "output": "Output Nodes",
        }
        header = header_map.get(category, "")
        if not header:
            return content

        lines = content.split("\n")
        start = None
        end = None

        for i, line in enumerate(lines):
            if header in line and line.startswith("#"):
                start = i
            elif start is not None and line.startswith("## ") and i > start:
                end = i
                break

        if start is not None:
            section = lines[start : end if end else len(lines)]
            return "\n".join(section)

        return content

    @staticmethod
    def _extract_pattern(content: str, pattern_name: str) -> str:
        """Extract a specific pattern from WORKFLOW_PATTERNS.md."""
        name_map = {
            "chatbot": "Pattern 1: Chatbot",
            "webhook_pipeline": "Pattern 2: Webhook Pipeline",
            "smart_router": "Pattern 3: Smart Router",
            "scheduled_task": "Pattern 4: Scheduled Task",
            "loop_processing": "Pattern 5: Loop Processing",
            "fan_out_fan_in": "Pattern 6: Fan-out / Fan-in",
        }
        section_header = name_map.get(pattern_name, "")
        if not section_header:
            return content

        lines = content.split("\n")
        start = None
        end = None

        for i, line in enumerate(lines):
            if section_header in line and line.startswith("#"):
                start = i
            elif start is not None and line.startswith("## Pattern") and i > start:
                end = i
                break

        if start is not None:
            section = lines[start : end if end else len(lines)]
            return "\n".join(section)

        return content
