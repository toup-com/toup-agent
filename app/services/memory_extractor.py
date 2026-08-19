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


# A6-2: seconds to wait before the single extraction retry. Module-level so
# tests can zero it out.
_EXTRACTION_RETRY_BACKOFF_S = 1.5


async def _complete_json_with_retry(llm, *, messages, temperature, max_tokens, model=None):
    """Call ``llm.complete_with_json`` with ONE retry after a short backoff.

    A6-2: per-turn fact extraction is a single un-fallbacked LLM call — a
    transient provider/network blip silently loses every fact stated that
    turn. One retry covers the transient class without meaningful cost;
    persistent failures (bad key, dead proxy) still fail fast on the second
    attempt, which propagates unchanged so the caller's no-regex-fallback
    policy stays intact.

    Returns (response, retried: bool).
    """
    import asyncio
    import logging

    try:
        response = await llm.complete_with_json(
            messages=messages, temperature=temperature, max_tokens=max_tokens,
            model=model,
        )
        return response, False
    except Exception as first_err:
        logging.warning(
            "[memory_extractor] LLM extraction attempt 1 failed, retrying "
            "once in %.1fs: %s: %s",
            _EXTRACTION_RETRY_BACKOFF_S,
            type(first_err).__name__, str(first_err)[:200],
        )
        await asyncio.sleep(_EXTRACTION_RETRY_BACKOFF_S)
        response = await llm.complete_with_json(
            messages=messages, temperature=temperature, max_tokens=max_tokens,
            model=model,
        )
        return response, True


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
    
    async def extract_memories_with_llm(
        self,
        user_message: str,
        assistant_response: str,
        brain_type: str = "user",
        max_memories: int = 15,
        api_key: Optional[str] = None,
        explicit_save_requested: bool = False,
    ) -> List[ExtractedMemory]:
        """
        Extract memories using LLM for more sophisticated understanding.
        Phase 4: Uses schema-enforced entity extraction with typed attributes.
        Falls back to rule-based extraction on failure.

        explicit_save_requested (D-mem-C, 2026-07-29): the caller detected
        explicit remember-phrasing in the user message ("please remember:",
        "for my records", "note this down"). Adds one rule to the prompt
        mandating verbatim capture of the requested fact — the base rules
        otherwise read a token-like payload ("memqa-…-lock-ece9") as a code
        snippet to SKIP — and relaxes the length/importance noise filters for
        this call only. The harness measured 5/8 explicit save requests
        dropped by exactly those two mechanisms.
        """
        from app.services.llm_service import get_llm_service, LLMService
        from app.services.extraction_schemas import generate_entity_schemas_prompt

        llm = LLMService(api_key=api_key) if api_key else get_llm_service()
        
        entity_schemas_doc = generate_entity_schemas_prompt()
        # Generated from app.memory_taxonomy so the prompt can never drift from
        # the enum again — this list used to be hand-maintained prose and had
        # diverged from BOTH enum copies.
        category_block = build_category_prompt_block()

        # D-mem-C: one extra numbered item, only on explicit-save turns, so
        # the base prompt stays byte-identical for every other call.
        explicit_save_block = ""
        if explicit_save_requested:
            explicit_save_block = """
12. **EXPLICITLY REQUESTED SAVE (ACTIVE THIS TURN)**: The user explicitly asked for something to be remembered ("remember this", "for my records", "note this down", "save this"). You MUST extract every fact the user asked to save, preserving its exact value VERBATIM — including token-like values, codes, passphrases, IDs and nicknames that would otherwise look like noise or code snippets (rule 2's code-snippet skip does NOT apply to them). Phrase each as a complete standalone sentence naming what the value is for, set importance to at least 0.8, and set transient to false unless the user gave an explicit expiry.
"""

        extraction_prompt = f"""You are a memory extraction system for a personal AI assistant. Your job is to extract information from this conversation that will still be worth knowing about THIS USER weeks from now.

USER MESSAGE:
{user_message}

ASSISTANT RESPONSE:
{assistant_response}

## What to Extract (be THOROUGH — extract ALL of these if present)

1. **Identity & biographical facts**: name, age, location, nationality, job, education, background
2. **Preferences & opinions**: likes, dislikes, favorites, strong opinions on any topic
3. **People mentioned**: names of friends, family, colleagues — and their relationship to the user
4. **Projects & work**: what they're building, working on, their role, their company
5. **Goals & plans**: short-term and long-term goals, aspirations, things they want to do
6. **Decisions made**: choices the user explicitly stated ("I decided to...", "I'm going with...")
7. **Skills & expertise**: technologies they use, languages they speak, tools they know
8. **Events & experiences**: things that happened, places visited, meetings attended
9. **Schedules & tasks**: upcoming deadlines, reminders, appointments, todos — these are almost always TRANSIENT (see rule 8)
10. **Relationships between entities**: "Alice works at Google", "Project X uses React", "My brother lives in Berlin"
11. **Corrections**: If the user corrects the agent ("No, actually...", "That's wrong, I meant...", "I didn't say that"), extract the CORRECT fact as a memory with high importance (0.9). Tag it with "correction" so the system can update or supersede the old incorrect memory.
{explicit_save_block}
## Entity Schema Types (IMPORTANT — use these for structured entity extraction)

{entity_schemas_doc}

## Extraction Rules (STRICT)

1. **Each memory MUST be a complete, standalone sentence** that makes sense without
   any surrounding context — **written in the second person, to the user** ("You…",
   "Your…"). Name other people by their name. NEVER write "The user…" or refer to
   the user in the third person.
   - GOOD: "You are applying to the UofT MScAC program for graduate studies"
   - GOOD: "Maya is your daughter and lives in Berlin"
   - BAD: "The user is applying to UofT" (third person) — or "Project: had" or "Fact: Can you check?"

2. **SKIP these entirely — do NOT extract:**
   - Greetings, pleasantries, conversational filler ("hi", "thanks", "sure", "ok")
   - Questions the user asked (those aren't facts to remember)
   - Fragments shorter than 5 words
   - The assistant's own suggestions or explanations (only extract USER facts)
   - Playback, search or tool OUTCOMES — a song that failed to play, the track that
     was played instead, an error the assistant hit. What the assistant did or
     couldn't do is never a fact about the user.
   - Momentary bodily states (hungry, tired, cold, sleepy) — at most a passing mood
     under rule 8, never a durable memory.
   - Vague or ambiguous statements that need context to understand
   - Technical commands or code snippets (unless they reveal a preference or decision)

3. **Only extract information STATED BY THE USER**, not inferred or from the assistant's response.

4. **Minimum quality bar:** If you read the memory 6 months from now with zero context, would it be useful and understandable? If not, don't extract it.

5. **Category must be EXACTLY one of these values** (use the definition to choose):
{category_block}

6. **Importance guide:**
   - 0.9-1.0: Core identity facts, major life decisions
   - 0.7-0.8: Strong preferences, active projects, goals
   - 0.5-0.6: Interesting facts, experiences, one-time events
   - 0.3-0.4: Minor preferences, casual mentions
   - Below 0.3: Don't bother extracting
   Importance measures how much this matters to the user LONG-TERM. Urgency is
   NOT importance: "urgently research X today" is a transient errand, not a
   core fact — mark it transient and score it low.

8. **Transience — decide this for EVERY memory.** Set `"transient": true` when the
   statement stops being true or stops being useful after a while:
   - Reminders, alarms, one-off errands ("remind me to eat tea in 2 minutes")
   - Requests scoped to today/this week ("research the 3 best PM tools for me")
   - Current status that will change ("I'm waiting on the deploy")
   - Passing moods
   Set `"transient": false` for durable facts: who they are, who they know, what
   they believe, what they can do, what they own, lasting preferences.
   When transient, add `"valid_for_days"` — your best estimate of how long it stays
   useful. A 2-minute reminder is 1; a this-week errand is 7; an active project
   status is 30. Omit `valid_for_days` when transient is false.
   A durable preference revealed BY a transient request should be extracted as its
   own separate, non-transient memory ("prefers tea in the afternoon").
   Also set `"scheduled": true` when the memory is a COMMITMENT THE USER HAS AT A
   SPECIFIC TIME — an appointment, flight, meeting, interview, deadline,
   reservation or booked call, whether or not a clock time is stated. Set it
   false for how they feel, where they currently are, and what they happen to be
   working on right now. "Dentist tomorrow at 3" is scheduled; "I'm exhausted
   today" and "I'm waiting on the deploy" are not. This is what keeps a
   short-lived appointment from being discarded as a passing mood, so decide it
   on every transient memory.

9. **Do NOT extract general world knowledge.** Facts about companies, shows, products
   or public figures that would be true for everybody are NOT memories about this user
   ("Anthropic develops Claude", "Better Call Saul is on Netflix"). Only keep such a
   fact when the user's own relationship to it is the point ("the user is watching
   Better Call Saul"). If you would not be surprised to find it in an encyclopedia,
   skip it.

7. **Schema-enforced entities**: For each entity, identify the best matching schema_type from the list above (PersonEntity, OrganizationEntity, ProjectEntity, PlaceEntity, EventEntity, TopicEntity, ToolEntity). Fill in as many fields as the conversation provides. If unsure, use "type" only (backward compatible).

Extract as many memories as the conversation warrants (up to {max_memories}). Do NOT artificially limit — if there are 10 distinct facts, extract all 10. Return ONLY valid JSON:
{{
  "memories": [
    {{
      "content": "Complete standalone sentence describing the memory",
      "category": "one of the valid category values listed above",
      "memory_type": "fact|preference|task|event|person|place|project|decision|skill",
      "importance": 0.7,
      "confidence": 0.9,
      "transient": false,
      "valid_for_days": null,
      "scheduled": false,
      "entities": [
        {{
          "name": "Alice",
          "type": "person",
          "schema_type": "PersonEntity",
          "data": {{"name": "Alice", "relationship_to_user": "friend", "occupation": "engineer", "organization": "Google"}}
        }},
        {{
          "name": "Google",
          "type": "organization",
          "schema_type": "OrganizationEntity",
          "data": {{"name": "Google", "org_type": "company", "industry": "technology"}}
        }}
      ],
      "tags": ["tag1", "tag2"]
    }}
  ]
}}

If the conversation is just casual chat, commands, or questions with nothing worth remembering long-term, return {{"memories": []}}. It is BETTER to extract nothing than to extract garbage."""

        # A6-2: outcome of this extraction, surfaced on the next turn's
        # [memory_health] line — "ok" / "retried" / "failed".
        self.last_extraction_outcome = None
        # Reasons the write-time gate rejected a proposed memory this turn.
        # Surfaced so an operator can tell "the model proposed nothing" apart
        # from "the model proposed six things and all six were junk" — the
        # two look identical from the outside and mean opposite things.
        _gated: List[str] = []
        _kept_notes: List[str] = []
        self.last_gated_reasons = _gated

        try:
            response, _retried = await _complete_json_with_retry(
                llm,
                messages=[{"role": "user", "content": extraction_prompt}],
                temperature=0.3,
                max_tokens=3000,
                # W1.4a: explicit pin — LLMService.default_model resolves to
                # the premium chat model when an Anthropic key is present.
                model=settings.memory_extraction_model,
            )

            # Parse the response — strip markdown fences if present (Anthropic)
            raw = response.content.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
            result = json.loads(raw)
            memories = []

            for mem_data in result.get("memories", [])[:max_memories]:
                content = mem_data.get("content", "").strip()

                # Quality filters — skip garbage. On an explicit-save turn the
                # length gates are relaxed (D-mem-C): the model is instructed
                # to write full sentences, but a short verbatim output like
                # "Parking code: memqa-x" must be kept, not silently dropped —
                # it is the exact fact the user just asked us to save.
                if not content:
                    continue
                if not explicit_save_requested:
                    if len(content) < 15:
                        continue
                    if content.count(" ") < 3:
                        continue
                if content.endswith("?"):
                    continue

                # Deterministic screen. Rule 9 of the prompt above already
                # says "Do NOT extract general world knowledge" and rule 3
                # says "Only extract information STATED BY THE USER" — both
                # were live on the fleet for twelve hours before the model
                # wrote five encyclopedia entries about 409A valuations,
                # restated from its own answer, into a user's brain. Prose
                # does not bind; this does.
                gate_reason = memory_gate_reason(
                    content,
                    user_message=user_message,
                    assistant_response=assistant_response,
                    # D-mem-C: the caller already detected "please remember: ..."
                    # phrasing. Without passing it through, the secret tier
                    # silently dropped the very facts the user asked to save —
                    # "remember my storage locker passphrase is kestrel-dbf7"
                    # was refused by a rule meant for observed credentials.
                    explicit_save=explicit_save_requested,
                )
                if gate_reason:
                    _gated.append(gate_reason)
                    continue

                # Canonicalise via the single taxonomy. Unknown values map to
                # `other` instead of the old `context` sink — `context` was in
                # no enum, so the app had no label for it and rendered "Other"
                # anyway, just without the value being queryable.
                # Kept as ENUM members, not plain strings. MemoryCategory /
                # MemoryType are str-subclasses, so `==` against a string still
                # works — but several consumers (api/chat.py:211,
                # modules/chat/router.py:210) do an unguarded `mem.category.value`,
                # which would raise AttributeError on a bare str. Returning the
                # enum keeps every existing consumer working.
                _category_str = normalize_category(
                    mem_data.get("category"), brain_type=brain_type
                )
                try:
                    category = MemoryCategory(_category_str)
                except ValueError:
                    # Non-user brain: the canonical value belongs to
                    # AgentCategory/WorkCategory, so it has no MemoryCategory
                    # member. The plain string is correct there; only the
                    # user-brain consumers do `.value`.
                    category = _category_str
                memory_type = MemoryType(
                    normalize_memory_type(mem_data.get("memory_type"))
                )

                try:
                    importance = float(mem_data.get("importance", 0.5))
                except (TypeError, ValueError):
                    importance = 0.5
                # A fact the user explicitly asked to save is never "not
                # worth extracting", whatever the model scored it (D-mem-C).
                if importance < 0.3 and not explicit_save_requested:
                    continue

                # Transience → expiry horizon. resolve_ttl_days refuses to
                # expire durable-fact categories even when the model says
                # transient, so a misclassification cannot cost a real fact.
                ttl_days = None
                # What the model ASKED for, before the never-expire override.
                # These are two different questions and conflating them was a
                # junk source: "I'm feeling queasy today" is flagged transient
                # with a 1-day horizon 8/8, but HEALTH never expires, so the
                # horizon became None and the passing state was stored forever.
                # Every durable health fact is flagged durable 8/8, so the
                # model's verdict separates them perfectly — the pipeline just
                # could not hear it.
                requested_ttl = None
                if bool(mem_data.get("transient")) and not describes_recurring_arrangement(content):
                    # A standing arrangement ("a Gmail briefing every day at
                    # 11:49") reads as a schedule and the model often flags it
                    # transient, but it is a durable preference — expiring it
                    # would stop the agent knowing about a routine the user
                    # still relies on.
                    ttl_days = resolve_ttl_days(
                        category, mem_data.get("valid_for_days")
                    )
                    requested_ttl = resolve_ttl_days(
                        category, mem_data.get("valid_for_days"),
                        respect_never_expire=False,
                    )

                # Second half of the transience decision: a horizon of a day or
                # less is conversational state, not memory, so it is not stored
                # at all rather than stored-then-expired. An explicitly
                # requested save is never dropped, whatever horizon the model
                # attached to it (D-mem-C).
                if not explicit_save_requested:
                    # Gate on the REQUESTED horizon; expire on the resolved
                    # one. The override exists so a TTL bug cannot delete a
                    # health fact — not so a passing state becomes permanent.
                    horizon_reason = transient_horizon_reason(
                        requested_ttl, category,
                        # ttl None + a requested horizon == a never-expire
                        # category: keeping this is a permanent decision.
                        permanent_if_kept=(ttl_days is None and requested_ttl is not None),
                    )
                    if horizon_reason in SCHEDULABLE_REASONS and is_scheduled_commitment(
                        content, bool(mem_data.get("scheduled"))
                    ):
                        # A commitment at a time is not a passing state. Keep it,
                        # and floor its TTL so the row outlives the appointment
                        # rather than expiring the morning of it.
                        # Never-expire categories keep ttl None; everything
                        # else gets a floor so the row outlives the commitment.
                        if ttl_days is not None:
                            ttl_days = scheduled_floor_ttl(ttl_days)
                        _kept_notes.append("scheduled_commitment")
                    elif horizon_reason:
                        _gated.append(horizon_reason)
                        continue

                # `summary` is deliberately NOT taken from the model. The
                # mobile card renders `summary || content`, so an LLM
                # abbreviation permanently outranked the full sentence — and
                # summary was never updated on edit or merge. Leaving it NULL
                # makes `content` the single rendered truth.
                memories.append(ExtractedMemory(
                    content=content,
                    summary=None,
                    category=category,
                    memory_type=memory_type,
                    importance=importance,
                    confidence=float(mem_data.get("confidence", 0.8) or 0.8),
                    entities=mem_data.get("entities", []),
                    tags=mem_data.get("tags", []),
                    metadata={"brain_type": brain_type, "extracted_by": "llm"},
                    ttl_days=ttl_days,
                ))

            self.last_extraction_outcome = "retried" if _retried else "ok"
            if _gated or _kept_notes:
                import logging
                logging.getLogger(__name__).info(
                    "[memory_gate] kept %d, rejected %d this turn: %s%s",
                    len(memories), len(_gated), ",".join(sorted(set(_gated))),
                    (" | kept-by: " + ",".join(sorted(set(_kept_notes)))) if _kept_notes else "",
                )
            return memories

        except Exception as e:
            # 2026-05-10 hardening: do NOT silently fall back to regex.
            # Regex extraction produces low-quality "Schedule: <fragment>"
            # output that pollutes the brain. The previous silent fallback
            # masked an LLM-call bug for weeks (max_tokens param mismatch
            # against gpt-5.x; see commit e0d004d). If LLM extraction
            # genuinely fails, we'd rather extract NOTHING this turn and
            # log loudly than store garbage.
            #
            # Operators can re-extract from past turns by calling the
            # backfill script once the underlying LLM issue is resolved.
            self.last_extraction_outcome = "failed"
            import logging
            logging.error(
                "[memory_extractor] LLM extraction failed (skipping turn — "
                "regex fallback disabled to keep brain clean): %s: %s",
                type(e).__name__, str(e)[:200],
            )
            return []
    
    async def extract_relationships_with_llm(
        self,
        user_message: str,
        assistant_response: str,
    ) -> List[Dict[str, Any]]:
        """
        Extract entity-entity relationships from conversation using LLM.
        Phase 4: Now includes properties dict for structured relationship data.
        
        Returns a list of relationship dicts:
        [{{"source": "Alice", "source_type": "person", 
          "target": "Google", "target_type": "organization",
          "relationship": "works_at", "confidence": 0.9,
          "properties": {{"role": "engineer", "since": "2023"}}}}]
        """
        from app.services.llm_service import get_llm_service

        llm = get_llm_service()

        # GENERATED from ENTITY_TYPE_TO_CATEGORY, not hand-listed. The previous
        # hardcoded list held 8 of the 20 types and no media types at all, so a
        # TV show could only be typed `topic` and a song `project` — which is
        # how category_for_relationship came to file "Better Call Saul is
        # available on Netflix" under Knowledge and "Drake artist of 0-100"
        # under Work.
        entity_type_block = build_entity_type_prompt_block()

        prompt = f"""Analyze this conversation and extract ALL entity-to-entity relationships mentioned by the user.

USER MESSAGE:
{user_message}

ASSISTANT RESPONSE:
{assistant_response}

## The test every relationship must pass
The USER, or someone/something in the USER's life, must be one end of the edge —
or the edge must describe a connection the user themselves asserted about their
own world. You are mapping THIS USER's world, not the world in general.

## What counts as a relationship:
- Person → Organization: "works at", "founded", "studies at"
- Person → Person: "is friend of", "is married to", "is sibling of", "manages"
- Person → Place: "lives in", "was born in", "visited"
- Person → Project: "works on", "created", "maintains"
- Project → Technology: "uses", "built with", "deployed on" (the user's project)

## NEVER extract (this is the most common mistake):
- General world knowledge true for everybody, regardless of who is asking:
  "Anthropic develops Claude", "Better Call Saul airs on Netflix",
  "Paris is in France". These are encyclopedia entries, not memories about
  the user. If the fact would be equally true for a stranger, DROP IT.
- Anything whose only source is the ASSISTANT RESPONSE. The assistant
  explaining a topic is not the user telling you about their life. Use the
  assistant response ONLY to resolve pronouns and names in the user's message.
- Relationships between two entities the user merely asked a question about.

## Rules:
- Only extract relationships explicitly stated by the USER
- Each entity must have a name and a type from this list, grouped by what it
  says about the user. Pick the MOST SPECIFIC type that fits — `topic` is the
  fallback for things that are genuinely just subjects, not a default:
  {entity_type_block}
- The relationship label should be a short verb phrase in snake_case
- Include any additional properties about the relationship (e.g. since, role, context)
- Confidence: 0.9+ for explicit statements, 0.6-0.8 for strong implications

Return ONLY valid JSON:
{{
  "relationships": [
    {{
      "source": "Alice",
      "source_type": "person",
      "target": "Google",
      "target_type": "organization",
      "relationship": "works_at",
      "confidence": 0.9,
      "properties": {{"role": "software engineer", "since": "2023"}}
    }}
  ]
}}

If no entity relationships are found, return {{"relationships": []}}.
Returning an empty list is the CORRECT answer for most conversations."""

        try:
            response = await llm.complete_with_json(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=1000,
                # W1.4a: explicit pin (see extract_memories_with_llm)
                model=settings.memory_extraction_model,
            )

            raw = response.content.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
            result = json.loads(raw)
            relationships = []

            for rel in result.get("relationships", []):
                source = rel.get("source", "").strip()
                target = rel.get("target", "").strip()
                relationship = rel.get("relationship", "").strip()

                if source and target and relationship and len(source) > 1 and len(target) > 1:
                    relationships.append({
                        "source": source,
                        "source_type": rel.get("source_type", "unknown"),
                        "target": target,
                        "target_type": rel.get("target_type", "unknown"),
                        "relationship": relationship,
                        "confidence": float(rel.get("confidence", 0.7)),
                        "properties": rel.get("properties", {}),
                    })

            return relationships

        except Exception as e:
            import logging
            logging.warning(f"LLM relationship extraction failed: {e}")
            return []

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
