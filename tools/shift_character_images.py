#!/usr/bin/env python3
"""Shift specified character crops vertically within a dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image


def shift_image(path: Path, delta: int) -> None:
    if delta == 0:
        return
    with Image.open(path) as img:
        arr = np.array(img.convert("L"))
    if delta > 0:
        arr = np.roll(arr, -delta, axis=0)
        arr[-delta:, :] = 255
    else:
        delta = abs(delta)
        arr = np.roll(arr, delta, axis=0)
        arr[:delta, :] = 255
    Image.fromarray(arr).save(path)


def process(root: Path, labels: List[str], delta: int) -> None:
    targets = set(labels)
    applied = 0
    for gt_path in sorted(root.glob("*.gt.txt")):
        label = gt_path.read_text(encoding="utf-8").strip()
        if label not in targets:
            continue
        tif_path = gt_path.with_suffix(".tif")
        if not tif_path.exists():
            continue
        shift_image(tif_path, delta)
        png_path = tif_path.with_suffix(".png")
        if png_path.exists():
            shift_image(png_path, delta)
        print(f"Shifted {gt_path.name} by {delta}px")
        applied += 1
    if applied == 0:
        print("No matching labels found.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shift selected character crops vertically.")
    parser.add_argument("--root", type=Path, required=True, help="Path to character dataset directory")
    parser.add_argument("--labels", type=str, nargs="+", help="Exact GT labels to shift")
    parser.add_argument("--pixels", type=int, required=True, help="Positive values move glyphs up.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process(args.root.resolve(), args.labels, args.pixels)


if __name__ == "__main__":
    main()
