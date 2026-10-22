#!/usr/bin/env bash
# A spoken question for `live_append_experiment.py --speech-file`: raw PCM16
# mono 24 kHz little-endian, no header.  macOS only (`say` + `afconvert`).
#
#   backend/scripts/make_speech_pcm.sh "What time does the library close tonight?"
#   OUT=/tmp/q3.pcm backend/scripts/make_speech_pcm.sh "Is it going to rain tomorrow?"
#
# Quote the sentence: in zsh an unquoted `?` is a glob and the call fails.
#
# Two traps, both checked on a real file (2026-10-05):
#   * `say` refuses LEI16 into an .aiff ("fmt?") — AIFF is big-endian — so it
#     writes BEI16 and afconvert makes it little-endian.
#   * afconvert's WAV header is NOT 44 bytes: a `FLLR` chunk pads it so the
#     samples start at offset 4096.  `tail -c +45` would put 4 KB of header —
#     a click — in front of the question, so the `data` chunk is located by
#     walking the RIFF chunks.
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <sentence...>   (OUT=<path>; default /tmp/toup-speech.pcm)" >&2
  exit 2
fi
out="${OUT:-/tmp/toup-speech.pcm}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

say -v Samantha -o "$work/x.aiff" --data-format=BEI16@24000 "$*"
afconvert -f WAVE -d LEI16@24000 -c 1 "$work/x.aiff" "$work/x.wav"

python3 - "$work/x.wav" "$out" <<'PY'
import struct
import sys

src, dst = sys.argv[1], sys.argv[2]
blob = open(src, "rb").read()
if blob[:4] != b"RIFF" or blob[8:12] != b"WAVE":
    sys.exit(f"{src}: not a RIFF/WAVE file")
fmt = data = None
pos = 12
while pos + 8 <= len(blob):
    cid, size = blob[pos:pos + 4], struct.unpack_from("<I", blob, pos + 4)[0]
    if cid == b"fmt ":
        fmt = struct.unpack_from("<HHIIHH", blob, pos + 8)
    elif cid == b"data":
        data = blob[pos + 8:pos + 8 + size]
        break
    pos += 8 + size + (size & 1)        # RIFF chunks are word-aligned
if fmt is None or data is None:
    sys.exit(f"{src}: no fmt/data chunk")
tag, channels, rate, _, _, bits = fmt
if (tag, channels, rate, bits) != (1, 1, 24000, 16):
    sys.exit(f"{src}: want PCM16 mono 24 kHz, got tag={tag} ch={channels} rate={rate} bits={bits}")
with open(dst, "wb") as fh:
    fh.write(data)
print(f"{dst}  {len(data) * 1000 // 48000} ms  ({len(data)} bytes, data at offset {pos + 8})")
PY
