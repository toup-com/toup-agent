"""
Voice Directives — Model can control voice, speed, and style in replies.

The model can embed directives in its response to control TTS output:
  [voice:alloy]       — Switch to a specific voice
  [speed:1.2]         — Adjust speech speed
  [pause:500]         — Insert a pause (ms)
  [emotion:excited]   — Set emotional tone
  [language:es]       — Switch language
  [whisper]           — Whisper mode
  [sing]              — Singing mode

The parser extracts these directives and returns clean text + directives.

Usage:
    from app.agent.voice_directives import parse_voice_directives

    text = "Hello [voice:nova] [speed:1.2] world!"
    result = parse_voice_directives(text)
    print(result.clean_text)   # "Hello  world!"
    print(result.directives)   # [VoiceDirective(type="voice", value="nova"), ...]
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# All recognized directive types
DIRECTIVE_TYPES = {
    "voice", "speed", "pause", "emotion", "language",
    "whisper", "sing", "model", "style",
}

# Valid voices (across providers)
VALID_VOICES = {
    "alloy", "echo", "fable", "onyx", "nova", "shimmer",  # OpenAI
    "rachel", "drew", "clyde", "paul", "domi", "bella",    # ElevenLabs
    "default",
}

# Valid emotions
VALID_EMOTIONS = {
    "neutral", "excited", "calm", "serious", "friendly",
    "dramatic", "whisper", "cheerful",
}


@dataclass
class VoiceDirective:
    """A single voice directive parsed from text."""
    type: str       # voice, speed, pause, emotion, etc.
    value: str      # The directive value
    position: int = 0  # Character position in original text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "value": self.value,
            "position": self.position,
        }


@dataclass
class DirectiveResult:
    """Result of parsing voice directives from text."""
    clean_text: str
    directives: List[VoiceDirective] = field(default_factory=list)
    voice: Optional[str] = None
    speed: float = 1.0
    emotion: Optional[str] = None
    language: Optional[str] = None
    pauses: List[int] = field(default_factory=list)  # Pause durations in ms
    whisper: bool = False
    sing: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clean_text": self.clean_text,
            "directives": [d.to_dict() for d in self.directives],
            "voice": self.voice,
            "speed": self.speed,
            "emotion": self.emotion,
            "language": self.language,
            "whisper": self.whisper,
            "sing": self.sing,
        }

    @property
    def has_directives(self) -> bool:
        return len(self.directives) > 0


# Regex patterns for directives
_DIRECTIVE_PATTERN = re.compile(
    r'\[('
    r'voice:[\w-]+'
    r'|speed:[\d.]+'
    r'|pause:\d+'
    r'|emotion:\w+'
    r'|language:\w+'
    r'|model:[\w.-]+'
    r'|style:\w+'
    r'|whisper'
    r'|sing'
    r')\]',
    re.IGNORECASE,
)


def parse_voice_directives(text: str) -> DirectiveResult:
    """
    Parse voice directives from text.

    Extracts [voice:X], [speed:X], [pause:X], [emotion:X], etc.
    and returns clean text with directives removed.

    Args:
        text: The raw text potentially containing directives.

    Returns:
        DirectiveResult with clean_text and parsed directives.
    """
    result = DirectiveResult(clean_text=text)
    directives = []

    for match in _DIRECTIVE_PATTERN.finditer(text):
        raw = match.group(1)
        pos = match.start()

        if ":" in raw:
            dtype, value = raw.split(":", 1)
            dtype = dtype.lower()
        else:
            dtype = raw.lower()
            value = "true"

        directive = VoiceDirective(type=dtype, value=value, position=pos)
        directives.append(directive)

        # Apply directive effects
        if dtype == "voice":
            result.voice = value.lower()
        elif dtype == "speed":
            try:
                result.speed = float(value)
                result.speed = max(0.25, min(4.0, result.speed))
            except ValueError:
                pass
        elif dtype == "pause":
            try:
                result.pauses.append(int(value))
            except ValueError:
                pass
        elif dtype == "emotion":
            result.emotion = value.lower()
        elif dtype == "language":
            result.language = value.lower()
        elif dtype == "whisper":
            result.whisper = True
        elif dtype == "sing":
            result.sing = True

    result.directives = directives

    # Remove directives from text
    result.clean_text = _DIRECTIVE_PATTERN.sub("", text).strip()
    # Clean up double spaces
    result.clean_text = re.sub(r'  +', ' ', result.clean_text)

    return result


def build_tts_config(result: DirectiveResult, defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Build a TTS configuration from parsed directives.

    Args:
        result: The parsed DirectiveResult.
        defaults: Default TTS config to merge with.

    Returns:
        Dict with voice, speed, model, and other TTS parameters.
    """
    config = dict(defaults or {})

    if result.voice:
        config["voice"] = result.voice
    if result.speed != 1.0:
        config["speed"] = result.speed
    if result.emotion:
        config["emotion"] = result.emotion
    if result.language:
        config["language"] = result.language
    if result.whisper:
        config["style"] = "whisper"
    if result.sing:
        config["style"] = "sing"

    return config


def validate_directive(dtype: str, value: str) -> tuple:
    """
    Validate a single directive.

    Returns:
        (is_valid, error_message)
    """
    if dtype not in DIRECTIVE_TYPES:
        return False, f"Unknown directive type: {dtype}"

    if dtype == "voice" and value.lower() not in VALID_VOICES:
        return False, f"Unknown voice: {value}. Valid: {', '.join(sorted(VALID_VOICES))}"

    if dtype == "speed":
        try:
            s = float(value)
            if s < 0.25 or s > 4.0:
                return False, "Speed must be 0.25-4.0"
        except ValueError:
            return False, "Speed must be a number"

    if dtype == "pause":
        try:
            p = int(value)
            if p < 0 or p > 10000:
                return False, "Pause must be 0-10000ms"
        except ValueError:
            return False, "Pause must be an integer (ms)"

    if dtype == "emotion" and value.lower() not in VALID_EMOTIONS:
        return False, f"Unknown emotion: {value}. Valid: {', '.join(sorted(VALID_EMOTIONS))}"

    return True, ""
