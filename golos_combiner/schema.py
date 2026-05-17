"""Parquet schema for the combined STT-benchmark dataset."""

from __future__ import annotations

import pyarrow as pa


def build_schema() -> pa.Schema:
    words_list = pa.list_(pa.struct([
        pa.field("text", pa.string()),
        pa.field("start", pa.float32()),
        pa.field("end", pa.float32()),
    ]))
    return pa.schema([
        pa.field("idx", pa.int32()),
        pa.field("audio_pcm", pa.binary()),
        pa.field("duration", pa.float32()),
        pa.field("words", words_list),
    ])
