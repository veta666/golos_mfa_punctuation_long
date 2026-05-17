"""Sanity-check verification for the combined parquet."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from .audio import pcm_duration


@dataclass
class VerifyConfig:
    parquet: Path
    tolerance: float = 0.05


def _is_monotonic(values: list[float]) -> bool:
    return all(values[i] <= values[i + 1] for i in range(len(values) - 1))


def run(cfg: VerifyConfig) -> int:
    pf = pq.ParquetFile(cfg.parquet)
    print("schema:")
    print(pf.schema_arrow)
    print(f"rows={pf.metadata.num_rows} row_groups={pf.num_row_groups}")

    row0 = pf.read_row_group(0).slice(0, 1).to_pylist()[0]
    print()
    print(f'row idx={row0["idx"]}  duration={row0["duration"]:.3f}s')
    print(f'  audio_pcm bytes: {len(row0["audio_pcm"])}')

    pcm_dur = pcm_duration(row0["audio_pcm"])
    print(
        f'  pcm-decoded duration: {pcm_dur:.3f}s  '
        f'(delta from duration field: {abs(pcm_dur - row0["duration"]) * 1000:.2f}ms)'
    )
    print(f'  words count: {len(row0["words"])}')
    print(f'  first 3 words: {row0["words"][:3]}')
    print(f'  last 3 words:  {row0["words"][-3:]}')
    starts = [w["start"] for w in row0["words"]]
    print(f"  starts monotonic: {_is_monotonic(starts)}")
    max_end = max((w["end"] for w in row0["words"]), default=0.0)
    print(
        f'  max word end:    {max_end:.3f}s  '
        f'(must be <= duration {row0["duration"]:.3f}s)'
    )

    print()
    print("cross-row sanity:")
    violations = 0
    for batch in pf.iter_batches(batch_size=200):
        for r in batch.to_pylist():
            max_end_r = max((w["end"] for w in r["words"]), default=0.0)
            if max_end_r > r["duration"] + cfg.tolerance:
                violations += 1
                if violations <= 3:
                    print(
                        f'  VIOLATION idx={r["idx"]}: '
                        f'max_end={max_end_r:.3f} > duration={r["duration"]:.3f}'
                    )
    print(f"  total violations: {violations}/{pf.metadata.num_rows}")
    return 1 if violations else 0
