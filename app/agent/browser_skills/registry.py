"""Skill registry — loads SKILL.md files and injects domain knowledge into system prompt."""

import logging
import os
from pathlib import Path
from typing import Dict, Optional

from .classifier import CategoryClassifier, ClassificationResult

logger = logging.getLogger(__name__)


class SkillRegistry:
    """Loads all browser skills at startup, classifies messages, injects skill prompts."""

    def __init__(self, skills_dir: Optional[str] = None):
        if skills_dir is None:
            skills_dir = str(Path(__file__).parent)
        self.skills_dir = skills_dir
        self.classifier = CategoryClassifier()
        self._skills: Dict[str, str] = {}  # category -> SKILL.md content
        self._load_all()

    def _load_all(self):
        """Load all SKILL.md files from subdirectories."""
        base = Path(self.skills_dir)
        for subdir in sorted(base.iterdir()):
            if not subdir.is_dir() or subdir.name.startswith(("_", ".")):
                continue
            skill_file = subdir / "SKILL.md"
            if skill_file.exists():
                content = skill_file.read_text(encoding="utf-8").strip()
                # Strip YAML frontmatter if present
                if content.startswith("---"):
                    end = content.find("---", 3)
                    if end != -1:
                        content = content[end + 3:].strip()
                self._skills[subdir.name] = content
                logger.info("[SKILLS] Loaded skill: %s (%d chars)", subdir.name, len(content))

        logger.info("[SKILLS] Loaded %d skills: %s", len(self._skills), list(self._skills.keys()))

    def classify(self, message: str) -> ClassificationResult:
        """Classify a user message into a category."""
        return self.classifier.classify(message)

    def get_skill_prompt(self, category: str) -> str:
        """Return the SKILL.md content for a category, or empty string if not found."""
        return self._skills.get(category, "")

    def get_skill_names(self) -> list:
        """Return list of available skill names."""
        return list(self._skills.keys())

    def build_augmented_prompt(self, base_prompt: str, message: str) -> tuple:
        """Classify message and return (augmented_system_prompt, classification_result).

        This is the main entry point used by ws_browser.py.
        """
        result = self.classify(message)

        if result.confidence < 0.4:
            # Low confidence — use general skill or base prompt only
            general = self._skills.get("general", "")
            if general:
                return base_prompt + "\n\n" + general, result
            return base_prompt, result

        skill_prompt = self.get_skill_prompt(result.category)
        if not skill_prompt:
            return base_prompt, result

        # Build the augmented prompt
        augmented = base_prompt + f"\n\n## DOMAIN SKILL: {result.category.upper().replace('_', ' ')}\n\n{skill_prompt}"

        # Add extracted parameters as context
        if result.params:
            param_lines = []
            for k, v in result.params.items():
                param_lines.append(f"- {k}: {v}")
            augmented += "\n\n## EXTRACTED PARAMETERS FROM USER MESSAGE:\n" + "\n".join(param_lines)

        return augmented, result
