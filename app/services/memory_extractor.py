"""
Memory extraction service - extracts structured memories from conversations.
Uses pattern matching and heuristics for entity and fact extraction.
Phase 4: Schema-enforced extraction with typed entities.
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from app.config import settings
from app.schemas import MemoryCategory, MemoryType
from app.services.memory_gate import (
    SCHEDULABLE_REASONS,
    is_scheduled_commitment,
    memory_gate_reason,
    scheduled_floor_ttl,
    transient_horizon_reason,
)
from app.memory_taxonomy import (
    build_category_prompt_block,
    build_entity_type_prompt_block,
    describes_recurring_arrangement,
    normalize_category,
    normalize_entity_type,
    normalize_memory_type,
    resolve_ttl_days,
)


@dataclass
class ExtractedMemory:
    """Represents an extracted memory from text.

    `category` / `memory_type` hold canonical taxonomy STRINGS (the enums are
    `str` subclasses, so existing callers that compared against enum members
    keep working). `summary` is normally None — see the extractor's note on why
    LLM-written summaries are no longer stored.
    """
    content: str
    summary: Optional[str]
    category: MemoryCategory
    memory_type: MemoryType
    importance: float
    confidence: float
    entities: List[Dict[str, Any]]  # [{"name": "...", "type": "...", "schema_type": "...", "data": {...}}]
    tags: List[str]
    metadata: Dict[str, Any]
    # None = never expires. Set when the model flags the memory transient and
    # the category permits expiry (app.memory_taxonomy.resolve_ttl_days).
    ttl_days: Optional[int] = None


@dataclass
class ExtractedEntity:
    """Represents an extracted entity"""
    name: str
    entity_type: str
    description: Optional[str] = None
    attributes: Optional[Dict[str, Any]] = None
    schema_type: Optional[str] = None


# A6-2's `_complete_json_with_retry` MOVED to `memory_curator` with the LLM
# call it protected. Nothing in this module reaches a model any more.


class MemoryExtractor:
    """
    Extracts structured memories from conversation text.
    Uses rule-based extraction with pattern matching.
    """
    
    # Patterns for different memory types
    PREFERENCE_PATTERNS = [
        r"(?:i|I)\s+(?:like|love|enjoy|prefer|hate|dislike|don't like)\s+(.+?)(?:\.|$|,)",
        r"(?:my|My)\s+favorite\s+(.+?)\s+is\s+(.+?)(?:\.|$)",
        r"(?:i|I)\s+(?:always|never|usually)\s+(.+?)(?:\.|$)",
    ]
    
    TASK_PATTERNS = [
        r"(?:i|I)\s+(?:need to|have to|should|must|want to|will)\s+(.+?)(?:\.|$)",
        r"(?:remind me to|don't forget to|todo:|task:)\s*(.+?)(?:\.|$)",
        r"(?:my|the)\s+goal\s+is\s+to\s+(.+?)(?:\.|$)",
    ]
    
    FACT_PATTERNS = [
        r"(?:i|I)\s+(?:am|work as|work at|live in|study at)\s+(.{5,80}?)(?:\.|$)",
        r"(?:my|My)\s+(?:name|job|age|birthday|email|phone)\s+is\s+(.{2,60}?)(?:\.|$)",
    ]
    
    EVENT_PATTERNS = [
        r"(?:yesterday|today|tomorrow|last week|next week|on \w+day)\s+(?:i|I|we)\s+(.+?)(?:\.|$)",
        r"(?:i|I)\s+(?:went|visited|attended|met|saw)\s+(.+?)(?:\.|$)",
        r"(?:in|on|at)\s+(\d{4}|\w+\s+\d+)\s*[,]?\s*(.+?)(?:\.|$)",
    ]
    
    PERSON_PATTERNS = [
        r"(?:my|My)\s+(?:friend|colleague|boss|partner|wife|husband|brother|sister|mother|father|son|daughter)\s+(\w+)",
        r"(\w+)\s+(?:is my|is a|works at|lives in)",
        r"(?:i|I)\s+(?:met|know|spoke with|talked to)\s+(\w+)",
    ]
    
    PROJECT_PATTERNS = [
        r"(?:i'm |i am |we're |we are )?(?:working on|building|creating|developing)\s+(?:a|an|the)\s+(.{10,80}?)(?:\.|$)",
        r"(?:project|app|application|website|system)\s+(?:called|named)\s+['\"]?([A-Z][\w\s]{2,30})['\"]?",
    ]
    
    HEALTH_PATTERNS = [
        r"(?:i|I)\s+(?:exercise|workout|run|jog|gym|swim|yoga)\s*(.+?)(?:\.|$|,)",
        r"(?:my|My)\s+(?:health|fitness|diet|weight|sleep)\s+(.+?)(?:\.|$)",
        r"(?:doctor|physician|medical|medicine|prescription|symptom)\s*(.+?)(?:\.|$)",
    ]
    
    FOOD_PATTERNS = [
        r"(?:i|I)\s+(?:eat|ate|cook|cooked|made)\s+(.+?)(?:\.|$)",
        r"(?:my|My)\s+favorite\s+(?:food|dish|meal|restaurant|cuisine)\s+(.+?)(?:\.|$)",
        r"(?:recipe|ingredient|cooking)\s+(.+?)(?:\.|$)",
    ]
    
    TRAVEL_PATTERNS = [
        r"(?:i|I)\s+(?:traveled|travelled|visited|went to)\s+(.+?)(?:\.|$)",
        r"(?:trip|vacation|holiday|flight|hotel)\s+(?:to|in)?\s*(.+?)(?:\.|$)",
        r"(?:planning to|want to|going to)\s+(?:visit|travel|go to)\s+(.+?)(?:\.|$)",
    ]
    
    LEARNING_PATTERNS = [
        r"(?:i|I)\s+(?:am learning|learned|studying|study)\s+(.+?)(?:\.|$)",
        r"(?:course|tutorial|class|lesson)\s+(?:on|about|in)?\s*(.+?)(?:\.|$)",
        r"(?:skill|technique|method)\s+(.+?)(?:\.|$)",
    ]
    
    SCHEDULE_PATTERNS = [
        r"(?:meeting|appointment|call|interview)\s+(?:at|on|with)?\s*(.+?)(?:\.|$)",
        r"(?:remind me|reminder|calendar)\s+(.+?)(?:\.|$)",
        r"(?:at|on)\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s+(.+?)(?:\.|$)",
    ]
    
    MEDIA_PATTERNS = [
        r"(?:book|movie|film|show|series|podcast|article)\s+(?:called|named|titled)?\s*['\"]?(.+?)['\"]?(?:\.|$)",
        r"(?:i|I)\s+(?:read|watch|watched|listened to|saw)\s+(.+?)(?:\.|$)",
        r"(?:author|director|artist|singer)\s+(\w+)",
    ]
    
    # Keywords for category classification
    CATEGORY_KEYWORDS = {
        MemoryCategory.IDENTITY: ["my name", "i am", "i'm", "years old", "born", "nationality", "background"],
        MemoryCategory.PREFERENCES: ["like", "love", "enjoy", "prefer", "hate", "dislike", "favorite",
                                     "food", "eat", "cook", "recipe", "restaurant", "meal", "cuisine"],
        MemoryCategory.BELIEFS: ["believe", "think", "opinion", "value", "important to me", "matters"],
        MemoryCategory.EMOTIONS: ["feel", "feeling", "happy", "sad", "angry", "excited", "anxious", "stressed"],
        # `family` keywords merged into PEOPLE, `food` into PREFERENCES and
        # `travel` into EXPERIENCES when the taxonomy was unified — those three
        # categories no longer exist as distinct values, and a naive remap
        # would have silently overwritten the surviving key's keyword list.
        MemoryCategory.PEOPLE: ["friend", "colleague", "coworker", "contact", "person", "met",
                                "family", "mother", "father", "brother", "sister", "wife",
                                "husband", "son", "daughter", "parent"],
        MemoryCategory.LOCATIONS: ["location", "address", "city", "country", "place", "where"],
        MemoryCategory.EXPERIENCES: ["happened", "remember", "yesterday", "last", "event", "experience",
                                     "travel", "trip", "vacation", "visit", "flight", "hotel", "destination"],
        MemoryCategory.ACTIVE_TASK: ["meeting", "appointment", "calendar", "remind", "schedule", "deadline"],
        MemoryCategory.WORK: ["work", "job", "office", "career", "professional", "company", "business",
                              "project", "build", "create", "develop", "working on", "system"],
        MemoryCategory.SKILLS: ["learn", "study", "course", "tutorial", "skill", "education"],
        MemoryCategory.KNOWLEDGE: ["know", "fact", "information", "definition", "meaning", "learned"],
        MemoryCategory.POSSESSIONS: ["tool", "software", "app", "application", "code", "programming", "config"],
        MemoryCategory.MEDIA: ["book", "movie", "film", "show", "series", "podcast", "music", "article"],
        MemoryCategory.HEALTH: ["health", "exercise", "fitness", "doctor", "medicine", "sleep", "diet"],
        MemoryCategory.HABITS: ["routine", "habit", "always", "every day", "usually", "ritual"],
        MemoryCategory.GOALS: ["goal", "plan", "want to", "aspire", "dream", "objective", "target"],
        MemoryCategory.OTHER: ["conversation", "chat", "discuss", "talk"],
    }
    
    # Entity type keywords
    # Keys MUST be canonical entity types (see ENTITY_TYPE_TO_CATEGORY) — a
    # guess is written straight to Entity.entity_type, and an invalid one is
    # not rejected, it silently resolves to Knowledge in
    # category_for_relationship. `date` was such a key: not a type at all.
    # Media keywords were absent entirely, which is the regex-path half of why
    # shows and songs never got a media type.
    ENTITY_KEYWORDS = {
        "person": ["he", "she", "they", "friend", "colleague", "boss", "family",
                   "brother", "sister", "wife", "husband", "partner", "manager"],
        "place": ["city", "country", "location", "address", "where", "at",
                  "restaurant", "cafe", "office", "mall", "store"],
        "organization": ["company", "corporation", "organization", "team", "group",
                         "startup", "employer", "client"],
        "project": ["project", "system", "product", "launch", "roadmap"],
        "software": ["app", "website", "platform", "service", "api", "dashboard"],
        "tool": ["tool", "library", "framework", "device", "cli"],
        "show": ["series", "season", "episode", "tv show", "sitcom"],
        "movie": ["movie", "film"],
        "music": ["song", "album", "track", "playlist", "artist", "band"],
        "book": ["book", "novel", "author", "chapter"],
        "event": ["monday", "tuesday", "wednesday", "thursday", "friday",
                  "saturday", "sunday", "january", "february", "meeting",
                  "appointment", "birthday", "deadline"],
        "skill": ["learning", "studying", "practising", "practicing", "course"],
    }
    
    def extract_memories(
        self,
        user_message: str,
        assistant_response: str,
        max_memories: int = 10
    ) -> List[ExtractedMemory]:
        """
        Extract memories from a conversation turn.
        Returns a list of structured memories.
        """
        memories = []
        combined_text = f"{user_message}\n{assistant_response}"
        
        # Extract preferences
        for pattern in self.PREFERENCE_PATTERNS:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            for match in matches[:2]:  # Limit per pattern
                content = match if isinstance(match, str) else " ".join(match)
                if not self._is_quality_content(content):
                    continue
                memories.append(self._create_memory(
                    content=f"User prefers: {content.strip()}",
                    original_text=content,
                    memory_type=MemoryType.PREFERENCE,
                    category=MemoryCategory.PREFERENCES,
                    importance=0.7
                ))
        
        # Extract tasks/goals
        for pattern in self.TASK_PATTERNS:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            for match in matches[:2]:
                content = match if isinstance(match, str) else " ".join(match)
                if not self._is_quality_content(content):
                    continue
                memories.append(self._create_memory(
                    content=f"Task: {content.strip()}",
                    original_text=content,
                    memory_type=MemoryType.TASK,
                    category=MemoryCategory.GOALS,
                    importance=0.8
                ))
        
        # Extract facts/identity info
        for pattern in self.FACT_PATTERNS:
            matches = re.findall(pattern, user_message, re.IGNORECASE)  # Focus on user's facts
            for match in matches[:2]:
                content = match if isinstance(match, str) else " ".join(match)
                if not self._is_quality_content(content):
                    continue
                memories.append(self._create_memory(
                    content=f"Fact: {content.strip()}",
                    original_text=content,
                    memory_type=MemoryType.FACT,
                    category=MemoryCategory.IDENTITY,
                    importance=0.6
                ))
        
        # Extract events/experiences
        for pattern in self.EVENT_PATTERNS:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            for match in matches[:2]:
                content = match if isinstance(match, str) else " ".join(match)
                if not self._is_quality_content(content):
                    continue
                memories.append(self._create_memory(
                    content=f"Event: {content.strip()}",
                    original_text=content,
                    memory_type=MemoryType.EVENT,
                    category=MemoryCategory.EXPERIENCES,
                    importance=0.5
                ))
        
        # Extract project mentions
        for pattern in self.PROJECT_PATTERNS:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            for match in matches[:1]:
                content = match if isinstance(match, str) else " ".join(match)
                if self._is_quality_content(content):
                    memories.append(self._create_memory(
                        content=f"Project: {content.strip()}",
                        original_text=content,
                        memory_type=MemoryType.PROJECT,
                        category=MemoryCategory.WORK,
                        importance=0.7
                    ))
        
        # Extract health mentions
        for pattern in self.HEALTH_PATTERNS:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            for match in matches[:1]:
                content = match if isinstance(match, str) else " ".join(match)
                if self._is_quality_content(content):
                    memories.append(self._create_memory(
                        content=f"Health: {content.strip()}",
                        original_text=content,
                        memory_type=MemoryType.NOTE,
                        category=MemoryCategory.HEALTH,
                        importance=0.6
                    ))
        
        # Extract food mentions
        for pattern in self.FOOD_PATTERNS:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            for match in matches[:1]:
                content = match if isinstance(match, str) else " ".join(match)
                if self._is_quality_content(content):
                    memories.append(self._create_memory(
                        content=f"Food: {content.strip()}",
                        original_text=content,
                        memory_type=MemoryType.PREFERENCE,
                        category=MemoryCategory.PREFERENCES,
                        importance=0.5
                    ))
        
        # Extract travel mentions
        for pattern in self.TRAVEL_PATTERNS:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            for match in matches[:1]:
                content = match if isinstance(match, str) else " ".join(match)
                if self._is_quality_content(content):
                    memories.append(self._create_memory(
                        content=f"Travel: {content.strip()}",
                        original_text=content,
                        memory_type=MemoryType.EVENT,
                        category=MemoryCategory.EXPERIENCES,
                        importance=0.6
                    ))
        
        # Extract learning mentions
        for pattern in self.LEARNING_PATTERNS:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            for match in matches[:1]:
                content = match if isinstance(match, str) else " ".join(match)
                if self._is_quality_content(content):
                    memories.append(self._create_memory(
                        content=f"Learning: {content.strip()}",
                        original_text=content,
                        memory_type=MemoryType.SKILL,
                        category=MemoryCategory.SKILLS,
                        importance=0.6
                    ))
        
        # Extract schedule/appointment mentions
        for pattern in self.SCHEDULE_PATTERNS:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            for match in matches[:1]:
                content = match if isinstance(match, str) else " ".join(match)
                if self._is_quality_content(content):
                    memories.append(self._create_memory(
                        content=f"Schedule: {content.strip()}",
                        original_text=content,
                        memory_type=MemoryType.TASK,
                        category=MemoryCategory.ACTIVE_TASK,
                        importance=0.8
                    ))
        
        # Extract media mentions
        for pattern in self.MEDIA_PATTERNS:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            for match in matches[:1]:
                content = match if isinstance(match, str) else " ".join(match)
                if self._is_quality_content(content):
                    memories.append(self._create_memory(
                        content=f"Media: {content.strip()}",
                        original_text=content,
                        memory_type=MemoryType.NOTE,
                        category=MemoryCategory.MEDIA,
                        importance=0.5
                    ))
        
        # Deduplicate and limit
        unique_memories = self._deduplicate_memories(memories)
        return unique_memories[:max_memories]
    
    def extract_entities(
        self,
        text: str
    ) -> List[ExtractedEntity]:
        """Extract named entities from text using pattern matching"""
        entities = []
        
        # Extract person names
        for pattern in self.PERSON_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                name = match.strip().title()
                if len(name) > 1 and name.isalpha():
                    entities.append(ExtractedEntity(
                        name=name,
                        entity_type="person"
                    ))
        
        # Simple capitalized word extraction for potential entities
        words = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
        seen_names = {e.name.lower() for e in entities}
        for word in words:
            if word.lower() not in seen_names and len(word) > 2:
                # Guess entity type based on context
                entity_type = self._guess_entity_type(word, text)
                if entity_type:
                    entities.append(ExtractedEntity(
                        name=word,
                        entity_type=entity_type
                    ))
                    seen_names.add(word.lower())
        
        return entities[:10]  # Limit entities
    
    def classify_category(self, text: str) -> MemoryCategory:
        """Classify text into a memory category based on keywords"""
        text_lower = text.lower()
        scores = {category: 0 for category in MemoryCategory}
        
        for category, keywords in self.CATEGORY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    scores[category] += 1
        
        # Get highest scoring category
        max_category = max(scores, key=scores.get)
        if scores[max_category] > 0:
            return max_category
        
        # Default to context for general information
        return MemoryCategory.OTHER
    
    # Alias for backward compatibility
    def classify_brain_region(self, text: str) -> MemoryCategory:
        return self.classify_category(text)
    
    def _create_memory(
        self,
        content: str,
        original_text: str,
        memory_type: MemoryType,
        category: MemoryCategory,
        importance: float
    ) -> ExtractedMemory:
        """Create an ExtractedMemory object"""
        # Extract entities from the content
        entities = [
            {"name": e.name, "type": e.entity_type}
            for e in self.extract_entities(original_text)
        ]
        
        # Generate tags from content
        tags = self._extract_tags(content)
        
        return ExtractedMemory(
            content=content,
            summary=content[:100] if len(content) > 100 else content,
            category=category,
            memory_type=memory_type,
            importance=importance,
            confidence=0.8,  # Rule-based extraction has decent confidence
            entities=entities,
            tags=tags,
            metadata={"source": "extraction", "original_text": original_text[:500]}
        )
    
    def _create_conversation_summary(
        self,
        user_message: str,
        assistant_response: str
    ) -> ExtractedMemory:
        """Create a conversation summary memory"""
        # Simple summarization: first 200 chars of user message
        summary = user_message[:200].strip()
        if len(user_message) > 200:
            summary += "..."
        
        category = self.classify_category(user_message)
        
        return ExtractedMemory(
            content=f"Conversation: {summary}",
            summary=summary[:100],
            category=MemoryCategory.OTHER,
            memory_type=MemoryType.CONVERSATION,
            importance=0.4,
            confidence=1.0,
            entities=self.extract_entities(user_message)[:5],
            tags=self._extract_tags(user_message),
            metadata={
                "source": "conversation",
                "user_message_length": len(user_message),
                "assistant_response_length": len(assistant_response)
            }
        )
    
    def _extract_tags(self, text: str) -> List[str]:
        """Extract relevant tags from text"""
        tags = set()
        text_lower = text.lower()
        
        # Add tags based on content type detection
        tag_keywords = {
            "work": ["work", "job", "office", "meeting", "colleague"],
            "personal": ["family", "friend", "home", "weekend"],
            "learning": ["learn", "study", "course", "tutorial"],
            "project": ["project", "build", "create", "develop"],
            "health": ["health", "exercise", "sleep", "diet"],
            "finance": ["money", "budget", "pay", "cost", "price"],
            "travel": ["travel", "trip", "visit", "flight", "hotel"],
            "tech": ["code", "programming", "software", "app", "computer"],
        }
        
        for tag, keywords in tag_keywords.items():
            if any(kw in text_lower for kw in keywords):
                tags.add(tag)
        
        return list(tags)[:5]
    
    def _guess_entity_type(self, name: str, context: str) -> Optional[str]:
        """Guess the type of an entity based on context"""
        context_lower = context.lower()
        name_lower = name.lower()
        
        # Check context around the name
        for entity_type, keywords in self.ENTITY_KEYWORDS.items():
            for kw in keywords:
                if kw in context_lower:
                    return entity_type
        
        # Default heuristics
        if name.endswith(("Inc", "Corp", "LLC", "Ltd")):
            return "organization"
        
        return None
    
    @staticmethod
    def _is_quality_content(content: str) -> bool:
        """Check if extracted content meets minimum quality bar.

        Must be strict — garbage memories are worse than no memories.
        """
        text = content.strip() if isinstance(content, str) else " ".join(content).strip()
        # Too short — meaningful facts need at least 10 chars
        if len(text) < 10:
            return False
        # Fewer than 3 words — can't be a meaningful sentence
        if text.count(" ") < 2:
            return False
        # Just a question — questions aren't facts to remember
        if text.endswith("?"):
            return False
        # Common garbage patterns (expanded)
        garbage = {
            "it", "that", "this", "yes", "no", "ok", "sure", "thanks",
            "hi", "hello", "hey", "can you", "please", "had", "the",
            "which", "what", "how", "who", "where", "when", "why",
            "something", "anything", "nothing", "new items", "into new items",
            "help and label", "memory and timeline", "component and app",
            "change something", "a something", "into something",
        }
        normalized = text.lower().strip().rstrip(".!,)]}")
        if normalized in garbage:
            return False
        # Reject if mostly non-alpha (code/syntax fragments)
        alpha_ratio = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
        if alpha_ratio < 0.5:
            return False
        # Reject fragments that look like assistant output, not user facts
        assistant_markers = [
            "here's", "here is", "i'll help", "let me", "i can help",
            "sure,", "of course", "certainly", "you can", "you should",
            "track your", "track it in", "track its",
        ]
        for marker in assistant_markers:
            if normalized.startswith(marker):
                return False
        return True

    def _deduplicate_memories(self, memories: List[ExtractedMemory]) -> List[ExtractedMemory]:
        """Remove duplicate or very similar memories"""
        seen_content = set()
        unique = []
        
        for memory in memories:
            # Normalize content for comparison
            normalized = memory.content.lower().strip()[:100]
            if normalized not in seen_content:
                seen_content.add(normalized)
                unique.append(memory)
        
        return unique
    
    # ── The LLM extractor is DELETED (v3 §2.1) ────────────────────────
    #
    # `extract_memories_with_llm` and `extract_relationships_with_llm` were
    # the primary and secondary producers of the row corpus, and both are
    # retired with it. The replacement is `memory_curator.curate_turn`,
    # which writes BULLETS INTO FILES rather than sentences into a table,
    # and is the one writer in the system.
    #
    # Their root defect is worth keeping written down, because it was not
    # in the prompt: the context block was literally
    # `USER MESSAGE:\n{user_message}`, and ws_chat hands the runner a
    # REWRITTEN user_message (a scraped YouTube title in a `[SYSTEM: …]`
    # line, Chrome page context, a reply quote). Every provenance rule
    # downstream — quoted / echo / unsupported — decided by measuring
    # overlap against that same string, so an injection into it disarmed
    # all three at once. The v3 writer is handed `display_user_message` and
    # told in the prompt that it is the only source of facts.
    #
    # What remains in this class is the RULE-BASED `extract_memories`, which
    # has two live consumers that are not the conversation path:
    # `app/api/ingest.py` (the external ingestion API) and
    # `app/scripts/seed_data.py`. Neither writes to the memory PRODUCT — the
    # rows they create are reachable only through `memory_search`'s document
    # leg — so they keep working unchanged.

    def _string_to_category(self, category_str: str) -> MemoryCategory:
        """Convert string to MemoryCategory enum.

        Delegates to the canonical normaliser so the alias table lives in one
        place (this used to be a fourth hand-maintained copy of the taxonomy).
        """
        return MemoryCategory(normalize_category(category_str))


    def _string_to_memory_type(self, type_str: str) -> MemoryType:
        """Convert string to MemoryType enum (canonical normaliser)."""
        return MemoryType(normalize_memory_type(type_str))


def get_memory_extractor() -> MemoryExtractor:
    """Get memory extractor instance"""
    return MemoryExtractor()
