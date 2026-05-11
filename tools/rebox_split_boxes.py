#!/usr/bin/env python3
"""
Rebox character images by splitting combined glyph labels into separate entries
using connected-component analysis.

This is useful for datasets where each .box line may contain multiple Baybayin
codepoints (base glyph + kudlit) but only one bounding box.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split combined glyph boxes via connected components.")
    parser.add_argument("--dir", type=Path, required=True, help="Directory containing .tif + .box files.")
    parser.add_argument("--threshold", type=int, default=250, help="Binarization threshold.")
    parser.add_argument("--margin", type=int, default=1, help="Expansion margin around each component.")
    return parser.parse_args()


def load_mask(path: Path, threshold: int) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("L")) < threshold


def connected_components(mask: np.ndarray, margin: int) -> Tuple[List[Tuple[int, int, int, int]], int]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps: List[Tuple[int, int, int, int]] = []
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            min_x = max_x = x
            min_y = max_y = y
            while stack:
                cy, cx = stack.pop()
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
                        min_x = min(min_x, nx)
                        max_x = max(max_x, nx)
                        min_y = min(min_y, ny)
                        max_y = max(max_y, ny)
            left = max(0, min_x - margin)
            right = min(w, max_x + 1 + margin)
            top = max(0, min_y - margin)
            bottom = min(h, max_y + 1 + margin)
            comps.append((left, bottom, right, top))

    comps.sort(key=lambda b: (b[1], b[0]))  # top-to-bottom, then left-to-right
    return comps, h


def tokens_from_box(box_path: Path) -> List[str]:
    tokens: List[str] = []
    for line in box_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        label = line.split()[0]
        tokens.extend(list(label))
    return tokens


def rebox_directory(directory: Path, threshold: int, margin: int) -> Tuple[int, int]:
    updated = 0
    skipped = 0
    for tif_path in sorted(directory.glob("*.tif")):
        box_path = tif_path.with_suffix(".box")
        if not box_path.exists():
            skipped += 1
            continue
        tokens = tokens_from_box(box_path)
        if not tokens:
            skipped += 1
            continue
        mask = load_mask(tif_path, threshold)
        comps, height = connected_components(mask, margin)
        if len(comps) != len(tokens):
            skipped += 1
            print(f"[WARN] {tif_path.stem}: tokens {len(tokens)} != components {len(comps)}")
            continue
        new_lines = []
        for token, (left, bottom, right, top) in zip(tokens, comps):
            new_lines.append(f"{token} {left} {height - bottom} {right} {height - top} 0")
        box_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        updated += 1
    return updated, skipped


def main() -> None:
    args = parse_args()
    directory = args.dir
    if not directory.exists():
        raise FileNotFoundError(directory)
    updated, skipped = rebox_directory(directory, args.threshold, args.margin)
    print(f"{directory}: updated {updated}, skipped {skipped}")


if __name__ == "__main__":
    main()
