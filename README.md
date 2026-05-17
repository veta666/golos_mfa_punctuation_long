---
pretty_name: Golos MFA Punctuation (Long)
language:
  - ru
license: other
task_categories:
  - automatic-speech-recognition
task_ids:
  - speech-recognition
size_categories:
  - 1K<n<10K
source_datasets:
  - extended|govnejri/golos_mfa_punctuation
tags:
  - russian
  - speech
  - stt
  - long-form
  - benchmark
  - word-timestamps
  - golos
dataset_info:
  features:
    - name: idx
      dtype: int32
    - name: audio
      dtype:
        audio:
          sampling_rate: 16000
    - name: duration
      dtype: float32
    - name: words
      sequence:
        - name: text
          dtype: string
        - name: start
          dtype: float32
        - name: end
          dtype: float32
  splits:
    - name: train
  config_name: default
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/golos_mfa_punctuation_long_00000.parquet
---

# Golos MFA Punctuation (Long)

Long-form Russian speech derived from
[`govnejri/golos_mfa_punctuation`](https://huggingface.co/datasets/govnejri/golos_mfa_punctuation).

## Purpose

Most public Russian STT corpora ship as short clips (a few seconds each).
For benchmarking long-form transcription, VAD, punctuation, and streaming
behavior, you want minutes-long audio with reliable word-level alignments.

This dataset builds those long clips by splicing groups of consecutive
short clips together, inserting randomized silences between them, and
shifting the original word timestamps onto the new combined timeline.

## Source

By default the build pipeline consumes the first shard of the source
dataset's `data/` directory:

```
data/train-00000-of-00034.parquet
```

Any shard with the same schema works — pass it via `--in-parquet`.

## Schema

| field      | type                                         | description                                                  |
|------------|----------------------------------------------|--------------------------------------------------------------|
| `idx`      | `int32`                                      | 1-based row index, matches `audio.path`'s number             |
| `audio`    | `struct<bytes: binary, path: string>`        | `bytes` is a full RIFF/WAVE file (16 kHz mono s16le, with header) holding the spliced clip — optionally `speechnorm`- and/or `loudnorm`-filtered when those flags are passed at build time. `path` is a synthetic label `combined{NNNN}.wav` (no file on disk). The parquet schema metadata declares this column as a Hugging Face `Audio` feature so the HF dataset viewer renders it as a playback widget. |
| `duration` | `float32`                                    | combined clip duration in seconds                            |
| `words`    | `list<{text: string, start: f32, end: f32}>` | word timings shifted onto the combined timeline              |

## How it is built

1. Read the source parquet (Opus-in-Ogg audio + JSON word list).
2. Walk the rows in order and accumulate them into groups of `--group`
   consecutive clips (default `15`).
3. Decode each clip to raw PCM with `ffmpeg`. Optional per-chunk filters
   (off by default): `--speechnorm` (per-segment gain riding for speech)
   and/or `--loudnorm` (EBU R128 integrated loudness target). When both
   are set, the chain runs `speechnorm,loudnorm`.
4. Concatenate the decoded PCM with `N-1` silence gaps. Gap lengths are
   drawn uniformly from `[--gap-min, --gap-max]` (default 0.05 s to 1.0 s)
   using a deterministic seed (`--seed`, default `42`).
5. Shift each clip's word `start`/`end` onto the combined timeline by the
   running offset accumulated from prior clips and silences.
6. Wrap the spliced PCM in a WAV header and write it to the `audio.bytes`
   field of one parquet row (with `audio.path = "combined{NNNN}.wav"`).

## Layout

```
golos_mfa_punctuation_long/
├── README.md                       # this file (HF dataset card)
├── golos_combiner/                 # library
│   ├── __init__.py
│   ├── audio.py                    # ffmpeg/Opus → PCM, PCM ↔ WAV, duration helpers
│   ├── schema.py                   # pyarrow schema
│   ├── assembly.py                 # per-group splice + timestamp shift
│   ├── builder.py                  # end-to-end build pipeline (BuildConfig + run)
│   └── verifier.py                 # parquet sanity check (VerifyConfig + run)
├── scripts/                        # thin CLI wrappers around the package
│   ├── build_combined_parquet.py
│   └── verify_combined_parquet.py
├── legacy/                         # original monolithic scripts (kept for reference)
│   ├── build_combined_parquet.py
│   └── verify_combined_parquet.py
└── data/                           # input shards + output parquet (not tracked)
    ├── train-00000-of-00034.parquet
    └── golos_mfa_punctuation_long_00000.parquet
```

## Requirements

- Python 3.10+
- `pyarrow`
- `ffmpeg` on `$PATH` (for Opus decoding and loudness normalization)

## Usage

### Build

```bash
# minimal — no normalization, output path auto-derived:
python scripts/build_combined_parquet.py \
  --in-parquet data/train-00000-of-00034.parquet
# writes data/golos_mfa_punctuation_long_00000.parquet

# any other shard:
python scripts/build_combined_parquet.py \
  --in-parquet data/train-00007-of-00034.parquet
# writes data/golos_mfa_punctuation_long_00007.parquet

# enable speech-leveling and broadcast loudness with default params:
python scripts/build_combined_parquet.py \
  --in-parquet data/train-00000-of-00034.parquet \
  --speechnorm \
  --loudnorm
# chain: speechnorm=p=0.9:c=1.5:e=1.5,loudnorm=I=-16:TP=-1.5:LRA=11

# fully explicit invocation with custom params:
python scripts/build_combined_parquet.py \
  --in-parquet data/train-00000-of-00034.parquet \
  --out-parquet data/golos_mfa_punctuation_long_00000.parquet \
  --group 15 \
  --gap-min 0.05 --gap-max 1.0 \
  --seed 42 \
  --workers 16 \
  --speechnorm "p=0.9:c=1.5:e=1.5" \
  --loudnorm   "I=-16:TP=-1.5:LRA=11"
```

Flags:

| flag                | default                                 | meaning                                                      |
|---------------------|-----------------------------------------|--------------------------------------------------------------|
| `--in-parquet`      | `data/train-00000-of-00034.parquet`     | source shard (any compatible shard works)                    |
| `--out-parquet`     | *auto-derived*                          | when omitted, derived from `--in-parquet` by matching `train-NNNNN-of-MMMMM` and producing `<dir>/golos_mfa_punctuation_long_NNNNN.parquet`. Hard-errors if the input name doesn't match the pattern. |
| `--group`           | `15`                                    | rows spliced per combined clip                               |
| `--gap-min`         | `0.05`                                  | minimum inter-clip silence (s)                               |
| `--gap-max`         | `1.0`                                   | maximum inter-clip silence (s)                               |
| `--seed`            | `42`                                    | RNG seed for reproducible gap sequence                       |
| `--workers`         | `16`                                    | parallel ffmpeg decoders per group                           |
| `--row-group-size`  | `50`                                    | combined rows flushed per parquet row group                  |
| `--keep-partial`    | off                                     | include the trailing group even if shorter than `--group`    |
| `--loudnorm`        | off                                     | ffmpeg `loudnorm` params (EBU R128 single-pass). Pass `--loudnorm` with no value to use defaults `I=-16:TP=-1.5:LRA=11`; pass `--loudnorm 'k=v:...'` to override. Omit to disable. |
| `--speechnorm`      | off                                     | ffmpeg `speechnorm` params. Pass `--speechnorm` with no value to use defaults `p=0.9:c=1.5:e=1.5`; pass `--speechnorm 'k=v:...'` to override. Omit to disable. Applied **before** `--loudnorm` when both are set. |

### Verify

```bash
python scripts/verify_combined_parquet.py --parquet data/golos_mfa_punctuation_long_00000.parquet
```

Checks the schema, prints first-row diagnostics, and scans every row for
`max(word.end) > duration` violations (within `--tolerance`, default 50 ms).

### From Python

```python
from pathlib import Path
from golos_combiner import BuildConfig, build, VerifyConfig, verify

build(BuildConfig(
    in_parquet=Path("data/train-00000-of-00034.parquet"),
    out_parquet=Path("data/golos_mfa_punctuation_long_00000.parquet"),
    group=15,
))
verify(VerifyConfig(parquet=Path("data/golos_mfa_punctuation_long_00000.parquet")))
```

### Load with `datasets`

```python
from datasets import load_dataset

ds = load_dataset(
    "parquet",
    data_files="data/golos_mfa_punctuation_long_00000.parquet",
    split="train",
)
row = ds[0]
audio = row["audio"]   # auto-decoded by the Audio feature
print(audio["array"].shape, audio["sampling_rate"], row["duration"], row["words"][:3])
```

The schema metadata embedded in the parquet declares `audio` as a Hugging
Face `Audio` feature, so `load_dataset` decodes the WAV bytes for you into
a NumPy array and exposes the sampling rate.

## Licensing

This repository contains derivative audio of the source dataset
`govnejri/golos_mfa_punctuation`; consult the source dataset card for its
licensing terms before redistribution.
