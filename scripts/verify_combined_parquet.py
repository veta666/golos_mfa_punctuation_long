#!/usr/bin/env python3
"""CLI entry point for verifying a combined parquet produced by build_combined_parquet.py."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from golos_combiner.verifier import VerifyConfig, run


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Sanity-check a combined parquet: schema, per-row WAV/duration "
            "agreement, monotonic word starts, and no word_end past duration."
        )
    )
    ap.add_argument("--parquet", type=Path, default=Path("data/golos_mfa_punctuation_long_00000.parquet"))
    ap.add_argument(
        "--tolerance", type=float, default=0.05,
        help="seconds of slack allowed for word_end > duration",
    )
    args = ap.parse_args()
    return run(VerifyConfig(parquet=args.parquet, tolerance=args.tolerance))


if __name__ == "__main__":
    sys.exit(main())
