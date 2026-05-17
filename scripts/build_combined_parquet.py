#!/usr/bin/env python3
"""CLI entry point for building the combined STT-benchmark parquet."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from golos_combiner.builder import BuildConfig, run

SHARD_RE = re.compile(r"train-(\d+)-of-\d+")


def derive_out_parquet(in_parquet: Path) -> Path | None:
    """Return the output path whose shard index matches the input's,
    or None if the input filename doesn't match `train-NNNNN-of-MMMMM`."""
    m = SHARD_RE.search(in_parquet.name)
    if not m:
        return None
    n = int(m.group(1))
    return in_parquet.parent / f"golos_mfa_punctuation_long_{n:05d}.parquet"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Group every N consecutive rows of a Golos parquet, splice them "
            "into one WAV-wrapped clip with randomized inter-clip silences, "
            "shift word timestamps onto the combined timeline, and write the "
            "result as an HF-compatible `audio` column."
        )
    )
    ap.add_argument(
        "--in-parquet", type=Path,
        default=Path("data/train-00000-of-00034.parquet"),
    )
    ap.add_argument(
        "--out-parquet", type=Path, default=None,
        help=(
            "output path. If omitted, derived from --in-parquet by matching "
            "'train-NNNNN-of-MMMMM' and producing "
            "'<dir>/golos_mfa_punctuation_long_NNNNN.parquet'. "
            "If the input name doesn't match that pattern, this flag is required."
        ),
    )
    ap.add_argument("--group", type=int, default=15)
    ap.add_argument("--gap-min", type=float, default=0.05)
    ap.add_argument("--gap-max", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument(
        "--row-group-size", type=int, default=50,
        help="flush this many combined rows per parquet row group",
    )
    ap.add_argument(
        "--keep-partial", action="store_true",
        help="include the trailing group even if shorter than --group",
    )
    ap.add_argument(
        "--loudnorm", nargs="?", metavar="PARAMS",
        const="I=-16:TP=-1.5:LRA=11", default=None,
        help=(
            "ffmpeg loudnorm filter params applied per-chunk before splicing "
            "(EBU R128 single-pass). Pass --loudnorm with no value to use "
            "defaults (I=-16:TP=-1.5:LRA=11 = broadcast targets). Pass "
            "--loudnorm 'k=v:...' to override. Omit the flag entirely to "
            "disable."
        ),
    )
    ap.add_argument(
        "--speechnorm", nargs="?", metavar="PARAMS",
        const="p=0.9:c=1.5:e=1.5", default=None,
        help=(
            "ffmpeg speechnorm filter params applied per-chunk before splicing. "
            "Pass --speechnorm with no value to use defaults "
            "(p=0.9:c=1.5:e=1.5). Pass --speechnorm 'k=v:...' to "
            "override. Omit the flag entirely to disable. Applied before "
            "--loudnorm when both are set."
        ),
    )
    args = ap.parse_args()

    out_parquet = args.out_parquet
    if out_parquet is None:
        out_parquet = derive_out_parquet(args.in_parquet)
        if out_parquet is None:
            print(
                f"cannot derive --out-parquet from {args.in_parquet.name!r}: "
                f"expected pattern 'train-NNNNN-of-MMMMM.parquet'. "
                f"Pass --out-parquet explicitly.",
                file=sys.stderr,
            )
            return 2

    return run(BuildConfig(
        in_parquet=args.in_parquet,
        out_parquet=out_parquet,
        group=args.group,
        gap_min=args.gap_min,
        gap_max=args.gap_max,
        seed=args.seed,
        workers=args.workers,
        row_group_size=args.row_group_size,
        keep_partial=args.keep_partial,
        loudnorm=args.loudnorm or None,
        speechnorm=args.speechnorm or None,
    ))


if __name__ == "__main__":
    sys.exit(main())
