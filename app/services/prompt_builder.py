"""
Prompt Builder Service - Constructs system prompts from identities, memories, and context

The prompt builder is responsible for:
1. Loading active identities (soul, user_profile, instructions, tools, context)
2. Formatting retrieved memories into context
3. Building the complete system prompt
4. Managing token budgets
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.db import Identity, Memory
from app.db.models import IdentityType
from app.services.llm_service import get_llm_service
from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class PromptContext:
    """Context used to build the prompt"""
    identities: List[Identity]
    memories: List[Dict[str, Any]]  # Memories with similarity scores
    history: List[Dict[str, str]]   # Recent conversation history
    system_prompt: str
    total_tokens: int


class PromptBuilder:
    """
    Builds system prompts for the chat agent.
    
    The system prompt is constructed in layers:
    1. Soul - Core personality and values
    2. User Profile - What we know about the user
    3. Agent Instructions - Behavioral guidelines
    4. Tools - Available capabilities
    5. Context - Dynamic situational context
    6. Memories - Relevant retrieved memories
    7. Current Time - For temporal awareness
    """
    
    # Token budget allocation (approximate)
    # Increased for GPT-5.2 / Claude Opus (200k context windows)
    MAX_SYSTEM_TOKENS = 12000
    MAX_MEMORY_TOKENS = 6000
    MAX_IDENTITY_TOKENS = 4000
    
    def __init__(self):
        self.llm_service = get_llm_service()
    
    async def build_system_prompt(
        self,
        user_id: str,
        db: AsyncSession,
        memories: Optional[List[Dict[str, Any]]] = None,
        include_time: bool = True
    ) -> PromptContext:
        """
        Build the complete system prompt for a user.
        
        Args:
            user_id: The user ID to build prompt for
            db: Database session
            memories: Optional list of retrieved memories with scores
            include_time: Whether to include current timestamp
        
        Returns:
            PromptContext with the built system prompt and metadata
        """
        # Load active identities ordered by priority
        identities = await self._load_identities(user_id, db)
        
        # Build prompt sections
        sections = []
        
        # 1. Soul (core personality)
        soul = self._get_identity_by_type(identities, IdentityType.SOUL)
        if soul:
            sections.append(f"# Core Identity\n{soul.content}")
        
        # 2. User Profile
        user_profile = self._get_identity_by_type(identities, IdentityType.USER_PROFILE)
        if user_profile:
            sections.append(f"# About the User\n{user_profile.content}")
        
        # 3. Agent Instructions
        instructions = self._get_identity_by_type(identities, IdentityType.AGENT_INSTRUCTIONS)
        if instructions:
            sections.append(f"# Behavioral Guidelines\n{instructions.content}")
        
        # 4. Tools
        tools = self._get_identity_by_type(identities, IdentityType.TOOLS)
        if tools:
            sections.append(f"# Available Tools\n{tools.content}")
        
        # 5. Dynamic Context
        context = self._get_identity_by_type(identities, IdentityType.CONTEXT)
        if context:
            sections.append(f"# Current Context\n{context.content}")
        
        # 6. Memories (if provided)
        if memories:
            memory_section = self._format_memories(memories)
            if memory_section:
                sections.append(memory_section)
        
        # 7. Current Time
        if include_time:
            now = datetime.utcnow()
            sections.append(f"# Current Time\n{now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        
        # Combine sections with token budget enforcement
        system_prompt = ""
        running_tokens = 0
        
        for section in sections:
            section_tokens = self.llm_service.count_tokens(section)
            if running_tokens + section_tokens > self.MAX_SYSTEM_TOKENS:
                # Truncate this section to fit remaining budget
                remaining_budget = self.MAX_SYSTEM_TOKENS - running_tokens
                if remaining_budget > 100:  # Only include if we have meaningful space
                    # Rough truncation: 1 token ≈ 4 chars
                    max_chars = remaining_budget * 4
                    truncated = section[:max_chars].rsplit("\n", 1)[0]  # Cut at last newline
                    system_prompt += "\n\n" + truncated if system_prompt else truncated
                break  # Stop adding sections
            
            system_prompt += "\n\n" + section if system_prompt else section
            running_tokens += section_tokens
        
        # Count tokens
        total_tokens = self.llm_service.count_tokens(system_prompt)
        
        return PromptContext(
            identities=identities,
            memories=memories or [],
            history=[],
            system_prompt=system_prompt,
            total_tokens=total_tokens
        )
    
    async def build_messages(
        self,
        user_id: str,
        db: AsyncSession,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        memories: Optional[List[Dict[str, Any]]] = None,
        max_history_messages: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """
        Build the complete message list for a chat completion.
        
        Args:
            user_id: User ID for identity loading
            db: Database session
            user_message: The current user message
            conversation_history: Previous messages in conversation
            memories: Retrieved memories to include
            max_history_messages: Maximum history messages to include
        
        Returns:
            List of message dicts ready for LLM
        """
        max_history = max_history_messages or settings.max_history_messages
        
        # Build system prompt
        context = await self.build_system_prompt(
            user_id=user_id,
            db=db,
            memories=memories,
            include_time=True
        )
        
        # Start with system message
        messages = [
            {"role": "system", "content": context.system_prompt}
        ]
        
        # Add conversation history (limited)
        if conversation_history:
            # Take last N messages
            recent_history = conversation_history[-max_history:]
            messages.extend(recent_history)
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    async def _load_identities(
        self,
        user_id: str,
        db: AsyncSession
    ) -> List[Identity]:
        """Load active identities for a user, ordered by priority."""
        query = (
            select(Identity)
            .where(
                and_(
                    Identity.user_id == user_id,
                    Identity.is_active == True
                )
            )
            .order_by(Identity.priority.desc())
        )
        
        result = await db.execute(query)
        return list(result.scalars().all())
    
    def _get_identity_by_type(
        self,
        identities: List[Identity],
        identity_type: IdentityType
    ) -> Optional[Identity]:
        """Get the highest priority identity of a specific type."""
        for identity in identities:
            if identity.identity_type == identity_type.value:
                return identity
        return None
    
    def _format_memories(
        self,
        memories: List[Dict[str, Any]],
        max_memories: int = 15
    ) -> str:
        """
        Format retrieved memories for inclusion in prompt.
        
        Memories are formatted with their relevance and content.
        Enforces MAX_MEMORY_TOKENS budget — stops adding memories when budget is exceeded.
        """
        if not memories:
            return ""
        
        lines = ["# Relevant Memories"]
        lines.append("The following memories may be relevant to this conversation:\n")
        
        header_text = "\n".join(lines)
        running_tokens = self.llm_service.count_tokens(header_text)
        
        included = 0
        for i, mem in enumerate(memories, 1):
            if included >= max_memories:
                break
            
            content = mem.get("content", "")
            category = mem.get("category", "general")
            score = mem.get("similarity_score", 0)
            brain_type = mem.get("brain_type", "user")
            
            # Format this memory entry
            entry = f"{included + 1}. [{brain_type}/{category}] (relevance: {score:.2f})\n   {content}\n"
            entry_tokens = self.llm_service.count_tokens(entry)
            
            # Check token budget before adding
            if running_tokens + entry_tokens > self.MAX_MEMORY_TOKENS:
                break  # Stop — budget exceeded
            
            lines.append(f"{included + 1}. [{brain_type}/{category}] (relevance: {score:.2f})")
            lines.append(f"   {content}")
            lines.append("")
            running_tokens += entry_tokens
            included += 1
        
        if included == 0:
            return ""
        
        return "\n".join(lines)
    
    def estimate_tokens(
        self,
        user_id: str,
        user_message: str,
        history_count: int = 0,
        memory_count: int = 0
    ) -> Dict[str, int]:
        """
        Estimate token usage for a request.
        
        Useful for planning and budget management.
        """
        # Base system prompt (without memories)
        base_system = 500  # Approximate base
        
        # User message
        user_tokens = self.llm_service.count_tokens(user_message)
        
        # Estimates
        history_tokens = history_count * 200  # ~200 tokens per message pair
        memory_tokens = memory_count * 100    # ~100 tokens per memory
        
        return {
            "system_estimate": base_system + memory_tokens,
            "user_message": user_tokens,
            "history_estimate": history_tokens,
            "total_estimate": base_system + memory_tokens + user_tokens + history_tokens
        }


# Singleton instance
_prompt_builder: Optional[PromptBuilder] = None


def get_prompt_builder() -> PromptBuilder:
    """Get the prompt builder singleton."""
    global _prompt_builder
    if _prompt_builder is None:
        _prompt_builder = PromptBuilder()
    return _prompt_builder
