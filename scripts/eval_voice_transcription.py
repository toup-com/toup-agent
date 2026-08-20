#!/usr/bin/env python3
"""Measure Farsi–English code-switching transcription accuracy, per config.

The 2026-08-16 voice session transcribed "grok bot" six ways — «سهش رو ببین
دراک باچی», «پاک پات», «راج راک», «آه گراک», «لاک», «گروه که ایلان ماسک» —
and the garbled rows fed the UI, the DB history every think path loads, and
memory extraction. Before changing the transcription config, this script puts
NUMBERS on the candidates; after, it is the regression harness. The house
rule applies: do not reason about this subsystem's numbers without running
the script.

What it does:
  1. Synthesizes a small utterance set (Farsi, English, and mid-sentence
     code-switched) via the TTS API — audio is cached beside this script.
  2. Runs each utterance through /v1/audio/transcriptions under each
     candidate config (model × prompt × language pin).
  3. Scores keyword recovery (did "Grok" survive as a recognizable token?)
     and SCRIPT DRIFT (did English audio come back in Persian script?), and
     prints every transcript so the numbers can be eyeballed.

Script drift is the round-ten failure and it is a different axis from keyword
recovery: a config can recover every keyword and still be unusable, because
the two consumers downstream read the SCRIPT, not the words. The relay's
reply-language directive flips the whole session to Persian on one Persian
transcript, and the app's `detectLang` latches its UI to Farsi on one Persian
sample — so a single drifted English utterance makes the agent answer an
English speaker in Farsi for the rest of the call. Drift on an `en` utterance
is therefore scored as a hard failure, not a quality score.

Usage:
    python3 scripts/eval_voice_transcription.py --env-file /path/to/.env
    (or with OPENAI_API_KEY already in the environment)

Costs a few cents per full run. TTS output is a proxy for a real speaker —
a native code-switcher's accent is harsher on the model than TTS is, so a
config that fails HERE will do worse on a phone; passing here is necessary,
not sufficient. The recordings stay the ground truth.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import unicodedata
import urllib.request
import uuid
from pathlib import Path

API = "https://api.openai.com/v1"
CACHE = Path(__file__).parent / ".voice-eval-cache"

# ── The utterance set ────────────────────────────────────────────────────
# Keywords are scored LOOSELY: any listed variant counts, so «گراک» (a fair
# Persian rendering of Grok) scores where «پاک پات» does not. The point is
# recoverability of the term, not romanization policy.
# `lang` is the language the audio is ACTUALLY in — the ground truth script
# drift is measured against. Unmarked entries are Farsi.
UTTERANCES = [
    {
        "id": "fa-grok-bot",
        "text": "در مورد grok bot یه توضیح کوتاه بده",
        "keywords": [["grok", "گراک", "گروک"]],
        "note": "the recording's core failure",
    },
    {
        "id": "fa-grok-musk",
        "text": "همون Grok رو میگم که ایلان ماسک ساخته",
        "keywords": [["grok", "گراک", "گروک"]],
        "note": "recording 1:12 — came out as «گروه که ایلان ماسک»",
    },
    {
        "id": "fa-products",
        "text": "فرق ChatGPT و Claude و Gemini دقیقا چیه؟",
        "keywords": [["chatgpt", "چت جی پی تی", "جی‌پی‌تی"], ["claude", "کلود"], ["gemini", "جمنای", "جمینی"]],
        "note": "three product names mid-Farsi",
    },
    {
        "id": "fa-only",
        "text": "یه آهنگ شاد برام پخش کن لطفا",
        "keywords": [["آهنگ"], ["پخش"]],
        "note": "Farsi control — a config must not regress this",
    },
    {
        "id": "en-only",
        "text": "Tell me about Grok bot, the assistant from xAI.",
        "keywords": [["grok"], ["xai", "x ai", "x.ai"]],
        "lang": "en",
        "note": "English control",
    },
    # ── The round-ten cases: English audio from a Persian-speaking account ──
    # This is the reported bug's exact input. The account resolves to a `fa`
    # hint from its own voice history (it is a Persian speaker), so every
    # config below that biases toward Persian is applied to THIS audio.
    {
        "id": "en-search",
        "text": "search me what is the strongest model for video generating?",
        "keywords": [["strongest"], ["video"]],
        "lang": "en",
        "note": "the 2026-08-20 recording, verbatim — answered in Farsi",
    },
    {
        "id": "en-short",
        "text": "what time is it?",
        "keywords": [["time"]],
        "lang": "en",
        "note": "short English — the case a per-utterance rule must still get right",
    },
    {
        "id": "en-accented-names",
        "text": "Play something by Ebi and then tell me the news.",
        "keywords": [["ebi", "ebbi"], ["news"]],
        "lang": "en",
        "note": "English carrying a Persian proper noun — must NOT flip the sentence",
    },
]

# Mirrors app.api.ws_realtime.transcription_prompt — keep the two in sync;
# the test suite pins the server side.
BIAS_TERMS = "Toup, Grok, ChatGPT, GPT, Claude, Gemini, xAI, OpenAI, Anthropic, YouTube, Spotify, WhatsApp, Telegram"
PROMPT_FA = (
    "گفت‌وگوی کاربر با یک دستیار هوشمند. فارسی محاوره‌ای، همراه با نام‌ها و "
    f"اصطلاح‌های انگلیسی مانند {BIAS_TERMS}."
)
PROMPT_EN = f"A user talking to an AI assistant. May mention: {BIAS_TERMS}."
# The round-ten candidate. PROMPT_FA asserts the audio IS colloquial Persian,
# which is a true statement about the SPEAKER and a false one about half their
# utterances — and the transcriber obeys the prompt over the audio. This one
# describes the repertoire instead and forbids translation outright, in both
# scripts so the model has both in context.
PROMPT_BI = (
    "A bilingual speaker talking to an AI assistant. They speak Persian "
    "(Farsi) and English, sometimes mixing them in one sentence. Transcribe "
    "each utterance in the language it was ACTUALLY spoken in — Persian "
    "speech in Persian script, English speech in English. Never translate.\n"
    "گویندهٔ دوزبانه است: فارسی محاوره‌ای و انگلیسی. هر جمله را دقیقاً به همان "
    "زبانی بنویس که گفته شده؛ هرگز ترجمه نکن.\n"
    f"May mention: {BIAS_TERMS}."
)

CONFIGS = [
    # (label, model, prompt, language)
    ("whisper-1 · pin=fa (V1 prod)", "whisper-1", None, "fa"),
    ("gpt-realtime-whisper · pin=fa (V2 prod)", "gpt-realtime-whisper", None, "fa"),
    ("gpt-4o-transcribe · bare", "gpt-4o-transcribe", None, None),
    ("gpt-4o-transcribe · promptFA (R9 prod)", "gpt-4o-transcribe", PROMPT_FA, None),
    ("gpt-4o-transcribe · promptBI (R10)", "gpt-4o-transcribe", PROMPT_BI, None),
    ("gpt-4o-transcribe · promptFA · pin=fa", "gpt-4o-transcribe", PROMPT_FA, "fa"),
]


def _key(env_file: str | None) -> str:
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    if env_file:
        for line in Path(env_file).read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"')
    sys.exit("No OPENAI_API_KEY in env and none found via --env-file.")


def _post_json(key: str, path: str, payload: dict) -> bytes:
    req = urllib.request.Request(
        API + path,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def tts(key: str, text: str, path: Path) -> None:
    if path.exists():
        return
    audio = _post_json(key, "/audio/speech", {
        "model": "gpt-4o-mini-tts",
        "voice": "alloy",
        "input": text,
        "response_format": "wav",
    })
    path.write_bytes(audio)


def transcribe(key: str, wav: bytes, model: str, prompt: str | None, language: str | None):
    boundary = uuid.uuid4().hex
    parts = io.BytesIO()

    def field(name: str, value: str):
        parts.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode())

    field("model", model)
    if prompt:
        field("prompt", prompt)
    if language:
        field("language", language)
    parts.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
                f"Content-Type: audio/wav\r\n\r\n".encode())
    parts.write(wav)
    parts.write(f"\r\n--{boundary}--\r\n".encode())
    req = urllib.request.Request(
        API + "/audio/transcriptions",
        data=parts.getvalue(),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["text"], None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read().decode()[:120]}"


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    return "".join(ch for ch in s if not unicodedata.category(ch).startswith("P"))


def score(transcript: str, keywords: list[list[str]]) -> tuple[int, int]:
    t = norm(transcript)
    hit = sum(1 for variants in keywords if any(norm(v) in t for v in variants))
    return hit, len(keywords)


# Mirrors app.api.voice.detect_script_language's Arabic-script range, which is
# what the relay and the app both actually test. Measured over LETTERS only,
# so punctuation and digits cannot dilute a wholly-Persian sentence.
_ARABIC = tuple(range(0x0600, 0x0700)) + tuple(range(0x0750, 0x0780)) + tuple(range(0xFB50, 0xFE00))
_ARABIC_SET = set(_ARABIC)


def persian_share(transcript: str) -> float:
    """Share of the transcript's LETTERS that are Arabic-script.

    This is the number both consumers key off: the relay flips the session's
    reply language when it clears its threshold, and the app latches its whole
    UI to Farsi on any of it at all. For English audio the correct value is 0.
    """
    letters = [c for c in transcript if c.isalpha()]
    if not letters:
        return 0.0
    fa = sum(1 for c in letters if ord(c) in _ARABIC_SET)
    return fa / len(letters)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-file", help="read OPENAI_API_KEY from this file")
    args = ap.parse_args()
    key = _key(args.env_file)

    CACHE.mkdir(exist_ok=True)
    for u in UTTERANCES:
        tts(key, u["text"], CACHE / f"{u['id']}.wav")
    print(f"audio ready in {CACHE}\n")

    totals: dict[str, list[int]] = {}
    drifts: dict[str, list[int]] = {}
    for label, model, prompt, language in CONFIGS:
        print(f"═══ {label} ═══")
        got = 0
        of = 0
        drifted = 0
        en_count = 0
        dead = False
        for u in UTTERANCES:
            wav = (CACHE / f"{u['id']}.wav").read_bytes()
            text, err = transcribe(key, wav, model, prompt, language)
            if err:
                print(f"  {u['id']}: UNSUPPORTED — {err}")
                dead = True
                break
            h, n = score(text, u["keywords"])
            got += h
            of += n
            # Drift is only meaningful where we know the audio's language.
            mark = ""
            if u.get("lang") == "en":
                en_count += 1
                share = persian_share(text)
                if share > 0:
                    drifted += 1
                    mark = f"  ⚠ DRIFT fa={share:.0%}"
            print(f"  {u['id']}: {h}/{n}  «{text.strip()}»{mark}")
        if not dead:
            totals[label] = [got, of]
            drifts[label] = [drifted, en_count]
            print(f"  → keyword recovery {got}/{of} · English drifted to Persian "
                  f"script {drifted}/{en_count}\n")
        else:
            print()

    print("═══ SUMMARY ═══")
    print("  drift  keywords  config")
    # Sorted by drift FIRST. A config that drifts is disqualified whatever its
    # keyword score: the downstream consumers read the script, not the words.
    for label in sorted(totals, key=lambda k: (drifts[k][0], -totals[k][0])):
        g, o = totals[label]
        d, e = drifts[label]
        flag = "  " if d == 0 else " ⚠"
        print(f"{flag} {d}/{e}    {g:2d}/{o}     {label}")


if __name__ == "__main__":
    main()
