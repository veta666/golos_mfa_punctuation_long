"""End-to-end build pipeline for the combined STT-benchmark parquet."""

from __future__ import annotations

import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .assembly import assemble_group
from .audio import FRAME_BYTES, FRAMERATE, decode_opus_to_pcm
from .schema import build_schema


@dataclass
class BuildConfig:
    in_parquet: Path
    out_parquet: Path
    group: int = 15
    gap_min: float = 0.05
    gap_max: float = 1.0
    seed: int = 42
    workers: int = 16
    row_group_size: int = 50
    keep_partial: bool = False
    loudnorm: str | None = None
    speechnorm: str | None = None


def run(cfg: BuildConfig) -> int:
    if cfg.gap_min < 0 or cfg.gap_max < cfg.gap_min:
        print("invalid --gap-min/--gap-max", file=sys.stderr)
        return 1

    max_silence = b"\x00" * (int(cfg.gap_max * FRAMERATE) * FRAME_BYTES)
    rng = random.Random(cfg.seed)
    schema = build_schema()

    pf = pq.ParquetFile(cfg.in_parquet)
    total_rows = pf.metadata.num_rows
    total_groups = (
        (total_rows + cfg.group - 1) // cfg.group
        if cfg.keep_partial
        else total_rows // cfg.group
    )
    print(
        f"in_rows={total_rows} group={cfg.group} target_groups={total_groups} "
        f"gap=[{cfg.gap_min},{cfg.gap_max}]s seed={cfg.seed} workers={cfg.workers} "
        f"speechnorm={cfg.speechnorm or 'off'} loudnorm={cfg.loudnorm or 'off'}",
        flush=True,
    )

    cfg.out_parquet.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(cfg.out_parquet, schema, compression="zstd")

    started = time.monotonic()
    total_gaps = 0
    gap_buckets: set[float] = set()
    buffer: list[dict] = []
    written = 0
    group_idx = 0
    pending: list[tuple[bytes, str]] = []

    def flush_buffer() -> None:
        nonlocal buffer, written
        if not buffer:
            return
        writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
        written += len(buffer)
        buffer = []
        elapsed = time.monotonic() - started
        rate = written / elapsed if elapsed else 0
        eta = (total_groups - written) / rate if rate else 0
        print(
            f"  {written}/{total_groups}  rate={rate:.1f}/s  eta={eta:.0f}s",
            flush=True,
        )

    def process_pending() -> None:
        nonlocal pending, group_idx, total_gaps
        group_idx += 1
        opus_bytes_list = [r[0] for r in pending]
        words_jsons = [r[1] for r in pending]
        pending = []
        with ThreadPoolExecutor(max_workers=min(cfg.workers, len(opus_bytes_list))) as ex:
            pcms = list(ex.map(
                lambda b: decode_opus_to_pcm(b, cfg.loudnorm, cfg.speechnorm),
                opus_bytes_list,
            ))
        gaps = [rng.uniform(cfg.gap_min, cfg.gap_max) for _ in range(len(pcms) - 1)]
        total_gaps += len(gaps)
        gap_buckets.update(round(g, 3) for g in gaps)
        buffer.append(assemble_group(group_idx, words_jsons, pcms, gaps, max_silence))
        if len(buffer) >= cfg.row_group_size:
            flush_buffer()

    try:
        for batch in pf.iter_batches(batch_size=512, columns=["audio", "words"]):
            bytes_arr = batch.column("audio").field("bytes")
            words_arr = batch.column("words")
            for i in range(len(bytes_arr)):
                pending.append((bytes_arr[i].as_py(), words_arr[i].as_py()))
                if len(pending) == cfg.group:
                    process_pending()
            if group_idx >= total_groups and not cfg.keep_partial:
                break

        if cfg.keep_partial and pending:
            process_pending()

        flush_buffer()
    finally:
        writer.close()

    elapsed = time.monotonic() - started
    print(
        f"done: wrote {written} rows to {cfg.out_parquet} in {elapsed:.1f}s; "
        f"{total_gaps} gaps, {len(gap_buckets)} distinct (ms-rounded)",
        flush=True,
    )
    return 0
