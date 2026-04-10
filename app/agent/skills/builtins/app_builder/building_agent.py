"""
Building Sub-Agent — Generates technical/implementation questions
with clickable options for the app builder pipeline.

Called internally by AppBuilderSkill after the Research Agent finishes (Layer 1B).
Receives research context + user's research answers to inform its questions.
The user never sees this agent directly — its output is presented by the main agent.
"""

import logging
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)


BUILDING_SYSTEM_PROMPT = """You are a Building Agent for an app builder platform that creates
cross-platform React Native/Expo apps (iPhone, iPad, Web).

Your job: Generate EXACTLY 5-7 technical/implementation questions with clickable [[option]] buttons.
These questions help determine HOW to build the app — UI, data, screens, interactions.

You will receive:
1. The user's app idea and chosen direction
2. Research findings and the user's answers to research questions
3. Context about what the app should include based on earlier conversation

Based on ALL of this context, generate questions covering:
- **UI/Visual style**: Color palette, dark/light mode, visual density
- **Key screens**: Which screens to include, navigation style
- **Data & storage**: What to persist, database needs, seed data
- **Interactions**: Main user actions, gestures, animations
- **Platform priority**: Mobile-first vs web-first, responsive breakpoints

IMPORTANT: Do NOT re-ask anything already answered in the research phase.
Build on what the user already decided — your questions go DEEPER into implementation.

These are PERSONAL apps for the user's own use — NOT commercial products.
NEVER ask about: ads, monetization, pricing, in-app purchases, subscriptions, analytics,
social sharing, push notifications for engagement, onboarding tours, user accounts/login,
app store metadata, or review prompts. None of these apply.

Output format — ONLY the questions, nothing else:
```
1. **[Technical question]**
[[Option A]] [[Option B]] [[Option C]]

2. **[Technical question]**
[[Option A]] [[Option B]]

...
```

RULES:
- Exactly 5-7 questions
- Every question MUST have [[option]] buttons on the NEXT LINE
- Do NOT include preamble, explanation, or closing text — ONLY numbered questions
- Keep questions SHORT (one line)
- Options should be specific and actionable (not vague)
- Reference what the user already decided when relevant
  (e.g., "Since you chose social features, how should the feed work?")
"""

BUILDING_USER_TEMPLATE = """App idea: "{original_idea}"
Chosen direction: "{chosen_direction}"

Research findings summary:
{research_context}

User's answers to research questions:
{research_answers}

Generate 5-7 technical/implementation questions with [[option]] buttons.
Build on what the user already decided — go deeper into UI, data, screens, and interactions."""


async def run_building_questions(
    original_idea: str,
    chosen_direction: str,
    research_context: str,
    research_answers: str,
    call_llm: Callable[..., Awaitable[str]],
) -> str:
    """
    Generate technical/implementation questions based on research context.

    Args:
        original_idea: The user's raw input
        chosen_direction: The direction from Layer 0
        research_context: The research agent's questions (what was asked)
        research_answers: The user's answers to research questions
        call_llm: Reference to AppBuilderSkill._call_llm

    Returns:
        Formatted string with 5-7 technical questions and [[option]] buttons
    """
    user_message = BUILDING_USER_TEMPLATE.format(
        original_idea=original_idea,
        chosen_direction=chosen_direction,
        research_context=research_context,
        research_answers=research_answers,
    )

    questions = await call_llm(
        system_prompt=BUILDING_SYSTEM_PROMPT,
        user_message=user_message,
        model="",  # dynamic — _call_llm picks best available provider
        purpose="building_agent_questions",
        max_tokens=4000,
    )

    return questions.strip()
