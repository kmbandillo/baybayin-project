#!/usr/bin/env python3
"""
Recompute bounding boxes for character TIFF/BOX sets so that each Baybayin codepoint
(including kudlit/pamudpod marks) receives its own precise rectangle.

Usage:
    python3 tools/rebox_char_components.py \
        --dir final_version/synthetic/characters/baybayin_namin \
        --dir final_version/synthetic/characters/tagalog_stylized
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
from PIL import Image


MARK_CHARS = {"ᜒ", "ᜓ", "᜔"}


@dataclass
class Component:
    left: int
    right: int
    top: int
    bottom: int
    area: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebox Baybayin characters via connected components.")
    parser.add_argument(
        "--dir",
        dest="dirs",
        type=Path,
        action="append",
        required=True,
        help="Directory containing paired .tif/.box/.gt.txt files.",
    )
    parser.add_argument("--threshold", type=int, default=250, help="Binary threshold (default: 250).")
    parser.add_argument("--margin", type=int, default=1, help="Extra pixels to expand each component box.")
    return parser.parse_args()


def load_binary(image_path: Path, threshold: int) -> np.ndarray:
    with Image.open(image_path) as img:
        arr = np.asarray(img.convert("L"))
    return arr < threshold


def connected_components(mask: np.ndarray, margin: int) -> List[Component]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: List[Component] = []
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            min_x = max_x = x
            min_y = max_y = y
            area = 0
            while stack:
                cy, cx = stack.pop()
                area += 1
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
            components.append(Component(left, right, top, bottom, area))
    return components


def to_tesseract_coords(comp: Component, height: int) -> tuple[int, int, int, int]:
    left = comp.left
    right = comp.right
    bottom = height - comp.bottom
    top = height - comp.top
    return left, bottom, right, top


def tokens_from_gt(gt_path: Path) -> List[str]:
    text = gt_path.read_text(encoding="utf-8").strip()
    return [ch for ch in text if not ch.isspace()]


def rebox_directory(directory: Path, threshold: int, margin: int) -> tuple[int, int]:
    updated = 0
    skipped = 0
    for tif_path in sorted(directory.glob("*.tif")):
        base = tif_path.stem
        box_path = tif_path.with_suffix(".box")
        gt_path = tif_path.with_suffix(".gt.txt")
        if not box_path.exists() or not gt_path.exists():
            skipped += 1
            continue

        tokens = tokens_from_gt(gt_path)
        mask = load_binary(tif_path, threshold)
        comps = connected_components(mask, margin)
        if len(tokens) != len(comps):
            skipped += 1
            print(f"[WARN] {base}: tokens {len(tokens)} != components {len(comps)}")
            continue

        comps_sorted = sorted(
            comps,
            key=lambda c: (-c.area, c.top, c.left),
        )
        height = mask.shape[0]
        new_lines = []
        for token, comp in zip(tokens, comps_sorted):
            left, bottom, right, top = to_tesseract_coords(comp, height)
            new_lines.append(f"{token} {left} {bottom} {right} {top} 0")
        box_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        updated += 1
    return updated, skipped


def main() -> None:
    args = parse_args()
    total_updated = 0
    total_skipped = 0
    for directory in args.dirs:
        if not directory.exists():
            print(f"[WARN] Missing directory: {directory}")
            continue
        updated, skipped = rebox_directory(directory, args.threshold, args.margin)
        total_updated += updated
        total_skipped += skipped
        print(f"{directory}: updated {updated}, skipped {skipped}")
    print(f"Total updated: {total_updated}, skipped: {total_skipped}")


if __name__ == "__main__":
    main()
