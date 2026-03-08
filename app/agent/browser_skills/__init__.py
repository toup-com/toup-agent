"""Browser skill system — category-based domain knowledge for the agentic browser."""

from .registry import SkillRegistry
from .classifier import CategoryClassifier, ClassificationResult

__all__ = ["SkillRegistry", "CategoryClassifier", "ClassificationResult"]
