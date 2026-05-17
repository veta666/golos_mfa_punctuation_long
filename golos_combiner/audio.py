"""Audio constants and PCM/WAV helpers used across the combiner pipeline."""

from __future__ import annotations

import io
import subprocess
import wave

FRAMERATE = 16000
SAMPWIDTH = 2
NCHANNELS = 1
FRAME_BYTES = NCHANNELS * SAMPWIDTH


def decode_opus_to_pcm(
    opus_bytes: bytes,
    loudnorm: str | None = None,
    speechnorm: str | None = None,
) -> bytes:
    """Decode Ogg/Opus to raw s16le PCM at FRAMERATE Hz, mono.

    Optional per-chunk filters, applied in this order when set:
      1. `speechnorm` — ffmpeg speechnorm (per-segment gain riding for speech),
         e.g. "p=0.9:c=1.5:e=1.5".
      2. `loudnorm` — ffmpeg single-pass EBU R128 integrated loudness target,
         e.g. "I=-16:TP=-1.5:LRA=11".

    Per-segment leveling first, then global loudness target.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-f", "ogg", "-i", "pipe:0",
    ]
    filters: list[str] = []
    if speechnorm:
        filters.append(f"speechnorm={speechnorm}")
    if loudnorm:
        filters.append(f"loudnorm={loudnorm}")
    if filters:
        cmd += ["-af", ",".join(filters)]
    cmd += ["-f", "s16le", "-ar", str(FRAMERATE), "-ac", str(NCHANNELS), "pipe:1"]
    proc = subprocess.run(cmd, input=opus_bytes, check=True, capture_output=True)
    return proc.stdout


def pcm_to_wav_bytes(pcm: bytes) -> bytes:
    """Wrap raw s16le mono PCM in a RIFF/WAVE container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(NCHANNELS)
        w.setsampwidth(SAMPWIDTH)
        w.setframerate(FRAMERATE)
        w.writeframes(pcm)
    return buf.getvalue()


def wav_duration(wav_bytes: bytes) -> float:
    """Return the duration in seconds of a RIFF/WAVE blob (header included)."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        return w.getnframes() / w.getframerate()


def pcm_duration(pcm: bytes) -> float:
    return len(pcm) / FRAME_BYTES / FRAMERATE
