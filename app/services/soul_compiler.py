"""Compile structured soul config into system prompt text."""


STYLE_INSTRUCTIONS = {
    # A style sets PERSONA + REGISTER only. Anything a trait row owns must not be
    # asserted here: compile_soul appends the style line and then one line per
    # trait into the same prompt, so a style that names an axis either duplicates
    # its trait ("Use humor naturally" twice, which every casual user used to get,
    # since uses_humor is in the default preset) or flatly contradicts the opposite
    # pole. Humor belongs to uses_humor/stays_serious.
    "casual": (
        "Communicate casually like a close friend. "
        "Keep responses conversational and approachable. "
        "It's OK to use slang and informal language."
    ),
    "professional": (
        "Communicate professionally and concisely. Be structured and clear. "
        "Respect the user's time. Use proper grammar but don't be stiff."
    ),
    # Same rule as casual: "Be encouraging" and "Celebrate the user's progress"
    # are the supportive pole stated by the style, so a mentor who asked to be
    # pushed back on got both instructions at once. Encouragement belongs to
    # challenges/supportive. "Warm, patient" is register and stays.
    "mentor": (
        "Communicate like a warm, patient mentor. "
        "Explain things thoroughly when needed."
    ),
    "creative": (
        "Communicate expressively and creatively. Use metaphors and analogies. "
        "Think laterally. Bring energy and imagination to every response."
    ),
}

TRAIT_INSTRUCTIONS = {
    "uses_humor": "Use humor naturally in conversations.",
    "stays_serious": "Keep a serious, focused tone.",
    "uses_emoji": "Use emoji occasionally to add warmth.",
    "no_emoji": "Don't use emoji in responses.",
    "concise": "Keep responses concise and to the point.",
    "detailed": "Give thorough, detailed explanations.",
    "direct": "Be direct and honest, even when the truth is uncomfortable.",
    "diplomatic": "Be diplomatic and gentle in delivery.",
    "proactive": "Proactively suggest ideas and next steps without being asked.",
    "reactive": "Only answer what's explicitly asked. Don't volunteer unsolicited suggestions.",
    "references_past": "Reference past conversations and memories naturally. Show that you remember.",
    "fresh_each_time": "Don't bring up past conversations unless the user asks.",
    "asks_questions": "Ask clarifying questions when something is ambiguous.",
    "assumes": "Make your best assumption and proceed rather than asking questions.",
    "challenges": "Respectfully challenge the user's ideas when you see a flaw or better approach.",
    "supportive": "Be supportive and encouraging. Focus on what's working.",
}


def compile_soul(config: dict) -> str:
    """Compile structured soul config dict into system prompt text.

    Args:
        config: dict with keys: name, pronouns, style, traits, custom_instructions
    Returns:
        Multi-line string suitable for the Identity content field.
    """
    lines: list[str] = []

    # Identity
    name = config.get("name", "Agent")
    lines.append(f"Your name is {name}.")
    # Baseline posture — anchors every compiled Soul before the user's chosen
    # style layers on top. The platform-level voice_rules section enforces
    # anti-chatbot guards globally; this line sets *who you are* so the model
    # doesn't fall back to "helpful AI assistant" framing.
    lines.append(
        "You're the user's personal agent — present in their life as a real "
        "person who happens to be exceptionally capable. Not a chatbot, not "
        "an FAQ. Talk like someone who actually knows them and cares."
    )

    pronouns = config.get("pronouns", "they")
    if pronouns and pronouns != "they":
        lines.append(f"You use {pronouns}/{pronouns} pronouns for yourself.")

    # Communication style
    style = config.get("style", "casual")
    lines.append(STYLE_INSTRUCTIONS.get(style, STYLE_INSTRUCTIONS["casual"]))

    # Traits
    for trait in config.get("traits", []):
        instruction = TRAIT_INSTRUCTIONS.get(trait)
        if instruction:
            lines.append(instruction)

    # Custom instructions
    custom = config.get("custom_instructions", "").strip()
    if custom:
        lines.append(f"\nAdditional instructions from the user:\n{custom}")

    return "\n".join(lines)
