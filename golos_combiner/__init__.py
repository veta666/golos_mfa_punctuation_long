"""Combined STT-benchmark parquet builder."""

from .assembly import assemble_group
from .audio import (
    FRAME_BYTES,
    FRAMERATE,
    NCHANNELS,
    SAMPWIDTH,
    decode_opus_to_pcm,
    pcm_duration,
)
from .builder import BuildConfig, run as build
from .schema import build_schema
from .verifier import VerifyConfig, run as verify

__all__ = [
    "FRAMERATE",
    "SAMPWIDTH",
    "NCHANNELS",
    "FRAME_BYTES",
    "decode_opus_to_pcm",
    "pcm_duration",
    "build_schema",
    "assemble_group",
    "BuildConfig",
    "VerifyConfig",
    "build",
    "verify",
]
