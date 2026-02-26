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

from app.schemas import MemoryCategory, MemoryType


@dataclass
class ExtractedMemory:
    """Represents an extracted memory from text"""
    content: str
    summary: str
    category: MemoryCategory
    memory_type: MemoryType
    importance: float
    confidence: float
    entities: List[Dict[str, Any]]  # [{"name": "...", "type": "...", "schema_type": "...", "data": {...}}]
    tags: List[str]
    metadata: Dict[str, Any]


@dataclass
class ExtractedEntity:
    """Represents an extracted entity"""
    name: str
    entity_type: str
    description: Optional[str] = None
    attributes: Optional[Dict[str, Any]] = None
    schema_type: Optional[str] = None


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
        r"(?:i|I)\s+(?:am|work as|work at|live in|study at)\s+(.+?)(?:\.|$)",
        r"(?:my|My)\s+(?:name|job|age|birthday|email|phone)\s+is\s+(.+?)(?:\.|$)",
        r"(.+?)\s+is\s+(?:a|an|the)\s+(.+?)(?:\.|$)",
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
        r"(?:working on|building|creating|developing)\s+(?:a|an|the)?\s*(.+?)(?:\.|$)",
        r"(?:project|app|application|website|system)\s+(?:called|named)?\s*['\"]?(\w+)['\"]?",
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
        MemoryCategory.PREFERENCES: ["like", "love", "enjoy", "prefer", "hate", "dislike", "favorite"],
        MemoryCategory.BELIEFS: ["believe", "think", "opinion", "value", "important to me", "matters"],
        MemoryCategory.EMOTIONS: ["feel", "feeling", "happy", "sad", "angry", "excited", "anxious", "stressed"],
        MemoryCategory.PEOPLE: ["friend", "colleague", "coworker", "contact", "person", "met"],
        MemoryCategory.PLACES: ["location", "address", "city", "country", "place", "where"],
        MemoryCategory.FAMILY: ["family", "mother", "father", "brother", "sister", "wife", "husband", "son", "daughter", "parent"],
        MemoryCategory.EXPERIENCES: ["happened", "remember", "yesterday", "last", "event", "experience"],
        MemoryCategory.PROJECTS: ["project", "build", "create", "develop", "working on", "app", "system"],
        MemoryCategory.SCHEDULE: ["meeting", "appointment", "calendar", "remind", "schedule", "at", "deadline"],
        MemoryCategory.WORK: ["work", "job", "office", "career", "professional", "company", "business"],
        MemoryCategory.LEARNING: ["learn", "study", "course", "tutorial", "skill", "education"],
        MemoryCategory.KNOWLEDGE: ["know", "fact", "information", "definition", "meaning", "learned"],
        MemoryCategory.TOOLS: ["tool", "software", "app", "application", "code", "programming", "config"],
        MemoryCategory.MEDIA: ["book", "movie", "film", "show", "series", "podcast", "music", "article"],
        MemoryCategory.HEALTH: ["health", "exercise", "fitness", "doctor", "medicine", "sleep", "diet"],
        MemoryCategory.HABITS: ["routine", "habit", "always", "every day", "usually", "ritual"],
        MemoryCategory.FOOD: ["food", "eat", "cook", "recipe", "restaurant", "meal", "cuisine"],
        MemoryCategory.TRAVEL: ["travel", "trip", "vacation", "visit", "flight", "hotel", "destination"],
        MemoryCategory.GOALS: ["goal", "plan", "want to", "aspire", "dream", "objective", "target"],
        MemoryCategory.CONTEXT: ["conversation", "chat", "discuss", "talk"],
    }
    
    # Entity type keywords
    ENTITY_KEYWORDS = {
        "person": ["he", "she", "they", "friend", "colleague", "boss", "family"],
        "place": ["city", "country", "location", "address", "where", "at"],
        "organization": ["company", "corporation", "organization", "team", "group"],
        "project": ["project", "app", "system", "product", "service"],
        "date": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "january", "february"],
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
                if len(content.strip()) > 2:
                    memories.append(self._create_memory(
                        content=f"Project: {content.strip()}",
                        original_text=content,
                        memory_type=MemoryType.PROJECT,
                        category=MemoryCategory.PROJECTS,
                        importance=0.7
                    ))
        
        # Extract health mentions
        for pattern in self.HEALTH_PATTERNS:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            for match in matches[:1]:
                content = match if isinstance(match, str) else " ".join(match)
                if len(content.strip()) > 2:
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
                if len(content.strip()) > 2:
                    memories.append(self._create_memory(
                        content=f"Food: {content.strip()}",
                        original_text=content,
                        memory_type=MemoryType.PREFERENCE,
                        category=MemoryCategory.FOOD,
                        importance=0.5
                    ))
        
        # Extract travel mentions
        for pattern in self.TRAVEL_PATTERNS:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            for match in matches[:1]:
                content = match if isinstance(match, str) else " ".join(match)
                if len(content.strip()) > 2:
                    memories.append(self._create_memory(
                        content=f"Travel: {content.strip()}",
                        original_text=content,
                        memory_type=MemoryType.EVENT,
                        category=MemoryCategory.TRAVEL,
                        importance=0.6
                    ))
        
        # Extract learning mentions
        for pattern in self.LEARNING_PATTERNS:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            for match in matches[:1]:
                content = match if isinstance(match, str) else " ".join(match)
                if len(content.strip()) > 2:
                    memories.append(self._create_memory(
                        content=f"Learning: {content.strip()}",
                        original_text=content,
                        memory_type=MemoryType.SKILL,
                        category=MemoryCategory.LEARNING,
                        importance=0.6
                    ))
        
        # Extract schedule/appointment mentions
        for pattern in self.SCHEDULE_PATTERNS:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            for match in matches[:1]:
                content = match if isinstance(match, str) else " ".join(match)
                if len(content.strip()) > 2:
                    memories.append(self._create_memory(
                        content=f"Schedule: {content.strip()}",
                        original_text=content,
                        memory_type=MemoryType.TASK,
                        category=MemoryCategory.SCHEDULE,
                        importance=0.8
                    ))
        
        # Extract media mentions
        for pattern in self.MEDIA_PATTERNS:
            matches = re.findall(pattern, combined_text, re.IGNORECASE)
            for match in matches[:1]:
                content = match if isinstance(match, str) else " ".join(match)
                if len(content.strip()) > 2:
                    memories.append(self._create_memory(
                        content=f"Media: {content.strip()}",
                        original_text=content,
                        memory_type=MemoryType.NOTE,
                        category=MemoryCategory.MEDIA,
                        importance=0.5
                    ))
        
        # Always create a conversation summary if we have content
        if len(user_message) > 20:
            summary = self._create_conversation_summary(user_message, assistant_response)
            memories.append(summary)
        
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
        return MemoryCategory.CONTEXT
    
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
            category=MemoryCategory.CONTEXT,
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
        """Check if extracted content meets minimum quality bar."""
        text = content.strip() if isinstance(content, str) else " ".join(content).strip()
        # Too short
        if len(text) < 5:
            return False
        # Fewer than 2 words
        if text.count(" ") < 1:
            return False
        # Just a question
        if text.endswith("?"):
            return False
        # Common garbage patterns
        garbage = {"it", "that", "this", "yes", "no", "ok", "sure", "thanks",
                    "hi", "hello", "hey", "can you", "please", "had", "the"}
        if text.lower().strip().rstrip(".!,") in garbage:
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
        max_memories: int = 15
    ) -> List[ExtractedMemory]:
        """
        Extract memories using LLM for more sophisticated understanding.
        Phase 4: Uses schema-enforced entity extraction with typed attributes.
        Falls back to rule-based extraction on failure.
        """
        from app.services.llm_service import get_llm_service
        from app.services.extraction_schemas import generate_entity_schemas_prompt
        
        llm = get_llm_service()
        
        entity_schemas_doc = generate_entity_schemas_prompt()
        
        extraction_prompt = f"""You are a memory extraction system for a personal AI assistant. Your job is to extract EVERY piece of noteworthy information from this conversation that the user would want their AI to remember permanently.

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
9. **Schedules & tasks**: upcoming deadlines, reminders, appointments, todos
10. **Relationships between entities**: "Alice works at Google", "Project X uses React", "My brother lives in Berlin"
11. **Corrections**: If the user corrects the agent ("No, actually...", "That's wrong, I meant...", "I didn't say that"), extract the CORRECT fact as a memory with high importance (0.9). Tag it with "correction" so the system can update or supersede the old incorrect memory.

## Entity Schema Types (IMPORTANT — use these for structured entity extraction)

{entity_schemas_doc}

## Extraction Rules (STRICT)

1. **Each memory MUST be a complete, standalone sentence** that makes sense without any surrounding context.
   - GOOD: "Nariman is applying to UofT MScAC program for graduate studies"
   - BAD: "Project: had" or "Fact: Can you check?" or "Task: it"

2. **SKIP these entirely — do NOT extract:**
   - Greetings, pleasantries, conversational filler ("hi", "thanks", "sure", "ok")
   - Questions the user asked (those aren't facts to remember)
   - Fragments shorter than 5 words
   - The assistant's own suggestions or explanations (only extract USER facts)
   - Vague or ambiguous statements that need context to understand
   - Technical commands or code snippets (unless they reveal a preference or decision)

3. **Only extract information STATED BY THE USER**, not inferred or from the assistant's response.

4. **Minimum quality bar:** If you read the memory 6 months from now with zero context, would it be useful and understandable? If not, don't extract it.

5. **Category must be one of:** identity, preferences, beliefs, emotions, people, places, family, experiences, projects, schedule, work, learning, knowledge, tools, media, health, habits, food, travel, goals, context

6. **Importance guide:**
   - 0.9-1.0: Core identity facts, major life decisions
   - 0.7-0.8: Strong preferences, active projects, goals
   - 0.5-0.6: Interesting facts, experiences, one-time events
   - 0.3-0.4: Minor preferences, casual mentions
   - Below 0.3: Don't bother extracting

7. **Schema-enforced entities**: For each entity, identify the best matching schema_type from the list above (PersonEntity, OrganizationEntity, ProjectEntity, PlaceEntity, EventEntity, TopicEntity, ToolEntity). Fill in as many fields as the conversation provides. If unsure, use "type" only (backward compatible).

Extract as many memories as the conversation warrants (up to {max_memories}). Do NOT artificially limit — if there are 10 distinct facts, extract all 10. Return ONLY valid JSON:
{{{{
  "memories": [
    {{{{
      "content": "Complete standalone sentence describing the memory",
      "summary": "Brief summary (max 100 chars)",
      "category": "one of the valid categories listed above",
      "memory_type": "fact|preference|task|event|person|place|project|decision|skill",
      "importance": 0.7,
      "confidence": 0.9,
      "entities": [
        {{{{
          "name": "Alice",
          "type": "person",
          "schema_type": "PersonEntity",
          "data": {{{{"name": "Alice", "relationship_to_user": "friend", "occupation": "engineer", "organization": "Google"}}}}
        }}}},
        {{{{
          "name": "Google",
          "type": "organization",
          "schema_type": "OrganizationEntity",
          "data": {{{{"name": "Google", "org_type": "company", "industry": "technology"}}}}
        }}}}
      ],
      "tags": ["tag1", "tag2"]
    }}}}
  ]
}}}}

If the conversation is just casual chat, commands, or questions with nothing worth remembering long-term, return {{{{"memories": []}}}}. It is BETTER to extract nothing than to extract garbage."""

        try:
            response = await llm.complete_with_json(
                messages=[{{"role": "user", "content": extraction_prompt}}],
                temperature=0.3,  # Lower temperature for more consistent extraction
                max_tokens=3000  # Increased to accommodate up to 15 memories
            )
            
            # Parse the response
            result = json.loads(response.content)
            memories = []
            
            # Valid categories for validation
            valid_categories = {{
                "identity", "preferences", "beliefs", "emotions", "people",
                "places", "family", "experiences", "projects", "schedule",
                "work", "learning", "knowledge", "tools", "media", "health",
                "habits", "food", "travel", "goals", "context"
            }}
            
            for mem_data in result.get("memories", [])[:max_memories]:
                content = mem_data.get("content", "").strip()
                
                # Quality filters — skip garbage
                if not content or len(content) < 15:
                    continue  # Too short to be meaningful
                if content.count(" ") < 3:
                    continue  # Fewer than 4 words
                if content.endswith("?"):
                    continue  # Questions aren't memories
                
                # Validate category
                category_str = mem_data.get("category", "context").lower()
                if category_str not in valid_categories:
                    category_str = "context"
                category = self._string_to_category(category_str)
                
                # Map memory type string to enum
                type_str = mem_data.get("memory_type", "fact").lower()
                memory_type = self._string_to_memory_type(type_str)
                
                importance = float(mem_data.get("importance", 0.5))
                if importance < 0.3:
                    continue  # Below minimum quality bar
                
                memories.append(ExtractedMemory(
                    content=content,
                    summary=mem_data.get("summary", "")[:100],
                    category=category,
                    memory_type=memory_type,
                    importance=importance,
                    confidence=float(mem_data.get("confidence", 0.8)),
                    entities=mem_data.get("entities", []),
                    tags=mem_data.get("tags", []),
                    metadata={{"brain_type": brain_type, "extracted_by": "llm"}}
                ))
            
            return memories
            
        except Exception as e:
            # Log error and fall back to rule-based extraction
            import logging
            logging.warning(f"LLM extraction failed, falling back to rules: {{e}}")
            return self.extract_memories(user_message, assistant_response, max_memories)
    
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
        
        prompt = f"""Analyze this conversation and extract ALL entity-to-entity relationships mentioned by the user.

USER MESSAGE:
{user_message}

ASSISTANT RESPONSE:
{assistant_response}

## What counts as a relationship:
- Person → Organization: "works at", "founded", "studies at"
- Person → Person: "is friend of", "is married to", "is sibling of", "manages"
- Person → Place: "lives in", "was born in", "visited"
- Person → Project: "works on", "created", "maintains"
- Project → Technology: "uses", "built with", "deployed on"
- Any meaningful connection between two named entities

## Rules:
- Only extract relationships explicitly stated or very strongly implied by the USER
- Each entity must have a name and type (person, organization, place, project, technology, event, topic, tool)
- The relationship label should be a short verb phrase in snake_case
- Include any additional properties about the relationship (e.g. since, role, context)
- Confidence: 0.9+ for explicit statements, 0.6-0.8 for strong implications

Return ONLY valid JSON:
{{{{
  "relationships": [
    {{{{
      "source": "Alice",
      "source_type": "person",
      "target": "Google",
      "target_type": "organization",
      "relationship": "works_at",
      "confidence": 0.9,
      "properties": {{{{"role": "software engineer", "since": "2023"}}}}
    }}}}
  ]
}}}}

If no entity relationships are found, return {{{{"relationships": []}}}}."""

        try:
            response = await llm.complete_with_json(
                messages=[{{"role": "user", "content": prompt}}],
                temperature=0.2,
                max_tokens=1000
            )
            
            result = json.loads(response.content)
            relationships = []
            
            for rel in result.get("relationships", []):
                source = rel.get("source", "").strip()
                target = rel.get("target", "").strip()
                relationship = rel.get("relationship", "").strip()
                
                if source and target and relationship and len(source) > 1 and len(target) > 1:
                    relationships.append({{
                        "source": source,
                        "source_type": rel.get("source_type", "unknown"),
                        "target": target,
                        "target_type": rel.get("target_type", "unknown"),
                        "relationship": relationship,
                        "confidence": float(rel.get("confidence", 0.7)),
                        "properties": rel.get("properties", {{}}),
                    }})
            
            return relationships
            
        except Exception as e:
            import logging
            logging.warning(f"LLM relationship extraction failed: {{e}}")
            return []

    def _string_to_category(self, category_str: str) -> MemoryCategory:
        """Convert string to MemoryCategory enum."""
        category_map = {
            "identity": MemoryCategory.IDENTITY,
            "preferences": MemoryCategory.PREFERENCES,
            "beliefs": MemoryCategory.BELIEFS,
            "emotions": MemoryCategory.EMOTIONS,
            "people": MemoryCategory.PEOPLE,
            "places": MemoryCategory.PLACES,
            "family": MemoryCategory.FAMILY,
            "experiences": MemoryCategory.EXPERIENCES,
            "projects": MemoryCategory.PROJECTS,
            "schedule": MemoryCategory.SCHEDULE,
            "work": MemoryCategory.WORK,
            "learning": MemoryCategory.LEARNING,
            "knowledge": MemoryCategory.KNOWLEDGE,
            "tools": MemoryCategory.TOOLS,
            "media": MemoryCategory.MEDIA,
            "health": MemoryCategory.HEALTH,
            "habits": MemoryCategory.HABITS,
            "food": MemoryCategory.FOOD,
            "travel": MemoryCategory.TRAVEL,
            "goals": MemoryCategory.GOALS,
            "context": MemoryCategory.CONTEXT,
        }
        return category_map.get(category_str.lower(), MemoryCategory.CONTEXT)
    
    def _string_to_memory_type(self, type_str: str) -> MemoryType:
        """Convert string to MemoryType enum."""
        type_map = {
            "fact": MemoryType.FACT,
            "preference": MemoryType.PREFERENCE,
            "task": MemoryType.TASK,
            "event": MemoryType.EVENT,
            "person": MemoryType.PERSON,
            "place": MemoryType.PLACE,
            "project": MemoryType.PROJECT,
            "decision": MemoryType.DECISION,
            "skill": MemoryType.SKILL,
            "file": MemoryType.FILE,
            "note": MemoryType.NOTE,
            "conversation": MemoryType.CONVERSATION,
        }
        return type_map.get(type_str.lower(), MemoryType.FACT)


def get_memory_extractor() -> MemoryExtractor:
    """Get memory extractor instance"""
    return MemoryExtractor()
