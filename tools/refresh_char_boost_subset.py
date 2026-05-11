#!/usr/bin/env python3
"""
Generate a fresh handwritten-character subset for Stage 2 training.

This script walks baybayin_dataset/handwritten/char_unbundled/,
samples a fixed number of .lstmf files per subdirectory, and writes
the absolute paths to the requested output file. Use a new --seed for
each refresh so the Stage 2 run sees different handwritten tiles.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--char-root",
        type=Path,
        default=Path("baybayin_dataset/handwritten/char_unbundled"),
        help="Root folder containing per-glyph subdirectories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination list file for the sampled .lstmf entries.",
    )
    parser.add_argument(
        "--sample-per-dir",
        type=int,
        default=40,
        help="Number of .lstmf files to sample per glyph folder.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. Use a new value per refresh for rotation.",
    )
    return parser


def collect_samples(char_root: Path, per_dir: int, seed: int | None) -> list[Path]:
    if seed is not None:
        random.seed(seed)
    if not char_root.is_dir():
        raise SystemExit(f"char root {char_root} does not exist or is not a directory")
    samples: list[Path] = []
    for sub in sorted(char_root.iterdir()):
        if not sub.is_dir():
            continue
        files = sorted(sub.rglob("*.lstmf"))
        if not files:
            continue
        if len(files) <= per_dir:
            chosen = files
        else:
            chosen = random.sample(files, per_dir)
        samples.extend(chosen)
    return samples


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    samples = collect_samples(args.char_root, args.sample_per_dir, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as fh:
        for path in samples:
            fh.write(str(path.resolve()) + "\n")
    print(
        f"Wrote {len(samples)} entries to {args.output} "
        f"(seed={args.seed}, per_dir={args.sample_per_dir})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
