"""Splice a group of decoded clips into one combined parquet record."""

from __future__ import annotations

import json

from .audio import FRAME_BYTES, FRAMERATE, pcm_duration


def assemble_group(
    idx: int,
    words_jsons: list[str],
    pcms: list[bytes],
    gaps: list[float],
    max_silence: bytes,
) -> dict:
    """Concatenate `pcms` with `gaps[i]` seconds of silence between consecutive
    clips, shift each clip's word timings onto the combined timeline, and
    return one parquet-ready record."""
    parts: list[bytes] = []
    words_out: list[dict] = []
    offset = 0.0
    last = len(pcms) - 1
    for i, (pcm, words_json) in enumerate(zip(pcms, words_jsons)):
        parts.append(pcm)
        try:
            clip_words = json.loads(words_json) if words_json else []
        except (TypeError, ValueError):
            clip_words = []
        for w in clip_words:
            words_out.append({
                "text": w["text"],
                "start": float(w["start"]) + offset,
                "end": float(w["end"]) + offset,
            })
        offset += pcm_duration(pcm)
        if i < last:
            silence_samples = int(gaps[i] * FRAMERATE)
            silence = max_silence[: silence_samples * FRAME_BYTES]
            parts.append(silence)
            offset += pcm_duration(silence)

    full_pcm = b"".join(parts)
    return {
        "idx": idx,
        "audio_pcm": full_pcm,
        "duration": float(pcm_duration(full_pcm)),
        "words": words_out,
    }
