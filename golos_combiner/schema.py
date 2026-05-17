"""Parquet schema for the combined STT-benchmark dataset.

The schema embeds a `huggingface` metadata blob declaring `audio` as an
Audio feature (sampling_rate=16000). This is what the HF dataset viewer
reads to render the audio playback widget.
"""

from __future__ import annotations

import json

import pyarrow as pa

SAMPLING_RATE = 16000

HF_FEATURES = {
    "info": {
        "features": {
            "idx": {"dtype": "int32", "_type": "Value"},
            "audio": {"sampling_rate": SAMPLING_RATE, "_type": "Audio"},
            "duration": {"dtype": "float32", "_type": "Value"},
            "words": {
                "feature": {
                    "text": {"dtype": "string", "_type": "Value"},
                    "start": {"dtype": "float32", "_type": "Value"},
                    "end": {"dtype": "float32", "_type": "Value"},
                },
                "_type": "Sequence",
            },
        }
    }
}


def build_schema() -> pa.Schema:
    audio_struct = pa.struct([
        pa.field("bytes", pa.binary()),
        pa.field("path", pa.string()),
    ])
    words_list = pa.list_(pa.struct([
        pa.field("text", pa.string()),
        pa.field("start", pa.float32()),
        pa.field("end", pa.float32()),
    ]))
    schema = pa.schema([
        pa.field("idx", pa.int32()),
        pa.field("audio", audio_struct),
        pa.field("duration", pa.float32()),
        pa.field("words", words_list),
    ])
    return schema.with_metadata({b"huggingface": json.dumps(HF_FEATURES).encode()})
