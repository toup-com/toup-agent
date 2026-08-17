"""The dispatch tone catalogue — ONE list, read by everything.

An operator picks the sound a dispatch arrives with. Three things then
have to agree about what "ping" means: the picker in the admin panel,
the validator on `POST /admin/dispatch`, and whatever eventually puts a
filename on an APNs payload. Two hand-maintained copies of that list is
how a panel comes to offer a tone the phone does not bundle — and the
symptom on device is the SYSTEM DEFAULT playing, i.e. exactly what a
correctly-chosen default sounds like. So there is one list, here, and
the panel renders `GET /admin/dispatch/tones` rather than a literal of
its own.

──────────────────────────────────────────────────────────────────────
WHERE A CHOSEN TONE CAN ACTUALLY SOUND — read this before extending
──────────────────────────────────────────────────────────────────────

iOS cannot be told to play an arbitrary system ringtone from a push. A
custom notification sound must be BUNDLED IN THE APP, <= 30 s, in
CAF/AIFF/WAV carrying Linear PCM / MA4 / uLaw / aLaw, and named by
FILENAME in the payload. Hence `file` below: it is a real resource in
`toup-platform-app/assets/sounds/`, installed into both the app target
and the LiveActivity widget target by `plugins/with-alarm-sound.js`.

And an admin dispatch's only push surface today is the Live Activity
push-to-start (`notification_dispatcher._dispatch_row` returns inside
the `_START_EVENT_KINDS` branch, before the Expo lane). iOS has NEVER
honoured a named sound on an ActivityKit push alert — the result is
SILENCE, not a fallback (`live_activity_service`, verified on device
2026-07-20/21 with the file in both bundles). So `apns_push` forces
"default" on every ActivityKit alert, and the tone chosen here reaches
the wire only on a standard alert push.

The consequence, stated plainly because it is the thing a reader will
otherwise assume away: **on today's shipped clients a chosen tone
changes nothing audible.** `docs/design/admin-dispatch-tones.md` names
the two prerequisites (a registered standard push token in the mobile
app, and an alert lane for `announcement` rows). Until those land, this
module's job is to make the choice EXIST, be validated, be persisted
and be carried — not to claim it rings.

──────────────────────────────────────────────────────────────────────
ANDROID
──────────────────────────────────────────────────────────────────────

Android binds a sound to a notification CHANNEL at channel-creation
time and it cannot be changed afterwards, so one tone == one channel.
`android_channel` is that channel's id — deterministic, derived from
the tone id, and DECLARED ONLY. No such channel exists on any device:
the app installs no `expo-notifications`, so it creates no channels and
registers no FCM token. Creating them needs a native module this repo
does not have. The id is fixed here anyway so that the day it lands,
the mapping is read from the same list as everything else.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# The system tone. Not a file — APNs spells it "default", and it is the
# only value iOS honours on a Live Activity alert.
TONE_DEFAULT = "default"


class Tone:
    """One catalogue entry.

    ``preview`` is a Web Audio recipe for the ADMIN PANEL, so an operator
    can hear roughly what they are choosing without shipping audio to the
    browser (the bundled files are CAF, which Chrome and Firefox do not
    decode). It is an approximation of
    ``toup-platform-app/scripts/gen-dispatch-tones.py``, which is the
    authority for what the phone plays; the two are kept in step by hand
    and that seam is documented in both repos' PRs.

    Each entry is ``[start_ms, freq_hz, dur_ms, timbre]`` — ``timbre`` is
    ``bell`` (inharmonic partials, long tail), ``blip`` (two partials,
    short) or ``thud`` (low + noisy).
    """

    __slots__ = ("id", "label", "description", "file", "android_channel", "preview")

    def __init__(
        self,
        id: str,
        label: str,
        description: str,
        file: Optional[str],
        preview: List[List[Any]],
    ) -> None:
        self.id = id
        self.label = label
        self.description = description
        self.file = file
        # A channel per tone, because Android cannot re-sound one channel.
        self.android_channel = "announcements" if file is None else f"announcements-{id}"
        self.preview = preview

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "file": self.file,
            "android_channel": self.android_channel,
            "preview": self.preview,
        }


# Ordered as the picker renders them: system default first, then the
# Toup tones from lightest to heaviest, critical last.
#
# The set is chosen for DISTINGUISHABILITY, not variety — six pleasant
# bells are one alert. Measured on the shipped files, every pair differs
# on register or on onset count:
#
#   tone     onsets   peak Hz   duration
#   ping        1       1759      0.32 s
#   chime       2       1568      1.00 s
#   knock       2        186      0.58 s
#   pulse       3        657      0.46 s
#   rise        4       1047      0.92 s
#   alarm      12        880      4.65 s
TONES: List[Tone] = [
    Tone(
        TONE_DEFAULT, "System default",
        "Whatever the phone plays for a normal notification.",
        None,
        [],
    ),
    Tone(
        "ping", "Ping",
        "One short, bright note. The lightest possible interruption.",
        "toup_ping.caf",
        [[0, 1760, 300, "blip"]],
    ),
    Tone(
        "chime", "Chime",
        "Two rising bells. Warm, and the closest to a message tone.",
        "toup_chime.caf",
        [[0, 1046.5, 600, "bell"], [200, 1568, 780, "bell"]],
    ),
    Tone(
        "pulse", "Pulse",
        "Three even taps at one pitch — legible through a pocket.",
        "toup_pulse.caf",
        [[0, 659.3, 110, "blip"], [90, 659.3, 110, "blip"], [180, 659.3, 110, "blip"]],
    ),
    Tone(
        "rise", "Rise",
        "Four ascending notes. The only tone here going somewhere — good news.",
        "toup_rise.caf",
        [[0, 523.25, 550, "blip"], [105, 659.25, 550, "blip"],
         [210, 783.99, 550, "blip"], [315, 1046.5, 550, "blip"]],
    ),
    Tone(
        "knock", "Knock",
        "Two low wooden thuds. Survives a noisy room, where a ping does not.",
        "toup_knock.caf",
        [[0, 184, 340, "thud"], [155, 184, 340, "thud"]],
    ),
    Tone(
        "alarm", "Alarm",
        "The reminder alarm — insistent, 4.6 s. Reserve it for something that "
        "cannot wait.",
        "toup_alarm.caf",
        [[0, 880, 180, "blip"], [230, 880, 180, "blip"], [460, 880, 180, "blip"],
         [690, 880, 180, "blip"]],
    ),
]

_BY_ID: Dict[str, Tone] = {t.id: t for t in TONES}

TONE_IDS = frozenset(_BY_ID)


def is_valid(tone_id: Optional[str]) -> bool:
    """None means 'the operator did not choose', which is the default."""
    return tone_id is None or tone_id in _BY_ID


def normalize(tone_id: Optional[str]) -> str:
    """Any unknown or missing id collapses to the system default.

    Read paths use this rather than raising: a tone id is cosmetic, and a
    dispatch that was accepted must still deliver if the catalogue is
    later trimmed. The WRITE path (`POST /admin/dispatch`) rejects instead
    — the operator has to be told their choice did not take.
    """
    return tone_id if tone_id in _BY_ID else TONE_DEFAULT


def apns_sound(tone_id: Optional[str]) -> str:
    """The value APNs wants in ``aps.alert.sound`` for this tone.

    A bundled filename, or "default". Note the caller does not get to
    decide whether the name survives: `apns_push` drops it on ActivityKit
    payloads, because a named sound there is silence.
    """
    tone = _BY_ID.get(normalize(tone_id))
    return tone.file if (tone and tone.file) else "default"


def catalogue() -> List[Dict[str, Any]]:
    return [t.as_dict() for t in TONES]
