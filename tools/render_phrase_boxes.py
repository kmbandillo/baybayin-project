#!/usr/bin/env python3
"""
Generate .tif/.box/.gt.txt files for phrase PNGs by separating base glyphs
and diacritics via connected components.

Usage:
    python3 tools/render_phrase_boxes.py --dir final_training_dataset/tagalog_stylized/ts_phrase
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Tesseract box files from phrase PNGs.")
    parser.add_argument("--dir", type=Path, required=True, help="Directory containing phrase PNG + .txt files.")
    parser.add_argument("--threshold", type=int, default=250, help="Binarization threshold (default: 250).")
    parser.add_argument("--margin", type=int, default=1, help="Extra pixels around each component.")
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

    comps.sort(key=lambda b: b[0])
    return comps, h


def tokens_from_txt(txt_path: Path) -> List[str]:
    text = txt_path.read_text(encoding="utf-8").strip()
    return [ch for ch in text if not ch.isspace()]


def process_directory(directory: Path, threshold: int, margin: int) -> Tuple[int, int]:
    updated = 0
    skipped = 0
    for png_path in sorted(directory.glob("*.png")):
        base = png_path.stem
        txt_path = directory / f"{base}.txt"
        if not txt_path.exists():
            skipped += 1
            continue

        tokens = tokens_from_txt(txt_path)
        if not tokens:
            skipped += 1
            continue

        mask = load_mask(png_path, threshold)
        comps, height = connected_components(mask, margin)
        if len(comps) != len(tokens):
            skipped += 1
            print(f"[WARN] {base}: tokens {len(tokens)} != components {len(comps)}")
            continue

        # Save TIFF version for Tesseract
        tif_path = directory / f"{base}.tif"
        with Image.open(png_path) as img:
            img.convert("L").save(tif_path)

        box_lines = []
        for token, (left, bottom, right, top) in zip(tokens, comps):
            box_lines.append(
                f"{token} {left} {height - bottom} {right} {height - top} 0"
            )
        (directory / f"{base}.box").write_text("\n".join(box_lines) + "\n", encoding="utf-8")
        (directory / f"{base}.gt.txt").write_text("".join(tokens) + "\n", encoding="utf-8")
        updated += 1
    return updated, skipped


def main() -> None:
    args = parse_args()
    directory = args.dir
    if not directory.exists():
        raise FileNotFoundError(directory)
    updated, skipped = process_directory(directory, args.threshold, args.margin)
    print(f"{directory}: generated {updated} files, skipped {skipped}")


if __name__ == "__main__":
    main()
