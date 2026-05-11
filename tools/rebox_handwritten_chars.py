#!/usr/bin/env python3
"""
Rebuild .box files for handwritten Baybayin samples so base glyphs and kudlit/pamudpod
marks get their own bounding boxes.

Usage:
    python rebox_handwritten_chars.py \
        --root full_dataset/final_finetune/mixed_finetune/character \
        --root full_dataset/final_finetune/mixed_finetune/words \
        --threshold 220 \
        --margin 2
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image


def load_binary(image_path: Path, threshold: int) -> np.ndarray:
    """Convert image to a binary mask (True = ink)."""
    img = Image.open(image_path).convert("L")
    arr = np.array(img)
    mask = arr < threshold
    return mask


def find_components(mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Return bounding boxes (left, top, right, bottom) for each 8-connected component."""
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    boxes: List[Tuple[int, int, int, int]] = []
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            queue = deque([(y, x)])
            visited[y, x] = True
            min_x = max_x = x
            min_y = max_y = y

            while queue:
                cy, cx = queue.pop()
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))
                        min_x = min(min_x, nx)
                        max_x = max(max_x, nx)
                        min_y = min(min_y, ny)
                        max_y = max(max_y, ny)

            boxes.append((min_x, min_y, max_x + 1, max_y + 1))

    boxes.sort(key=lambda b: (b[0], b[1]))
    return boxes


def expand_box(box: Tuple[int, int, int, int], width: int, height: int, margin: int) -> Tuple[int, int, int, int]:
    """Expand box by a uniform margin, clipped to image bounds."""
    left, top, right, bottom = box
    left = max(0, left - margin)
    top = max(0, top - margin)
    right = min(width, right + margin)
    bottom = min(height, bottom + margin)
    return left, top, right, bottom


def load_gt_tokens(gt_path: Path) -> List[str]:
    """Load GT text and return a list of codepoints, skipping spaces and newlines."""
    text = gt_path.read_text(encoding="utf-8")
    tokens: List[str] = []
    for ch in text:
        if ch in {" ", "\n", "\r", "\t"}:
            continue
        tokens.append(ch)
    return tokens


def write_box(path: Path, tokens: Sequence[str], boxes: Sequence[Tuple[int, int, int, int]], height: int) -> None:
    """Write Tesseract .box file in “char left bottom right top page” format."""
    with path.open("w", encoding="utf-8") as fh:
        for ch, (left, top, right, bottom) in zip(tokens, boxes):
            bottom_inv = height - bottom
            top_inv = height - top
            fh.write(f"{ch} {left} {bottom_inv} {right} {top_inv} 0\n")


def process_image(image_path: Path, threshold: int, margin: int) -> Tuple[bool, str]:
    gt_path = image_path.with_suffix(".gt.txt")
    if not gt_path.exists():
        return False, "missing_gt"

    tokens = load_gt_tokens(gt_path)
    if not tokens:
        return False, "empty_gt"

    mask = load_binary(image_path, threshold)
    height, width = mask.shape
    components = find_components(mask)
    if len(components) != len(tokens):
        return False, f"token_component_mismatch:{len(tokens)}:{len(components)}"

    expanded = [expand_box(box, width, height, margin) for box in components]
    box_path = image_path.with_suffix(".box")
    write_box(box_path, tokens, expanded, height)
    return True, ""


def iter_images(root: Path) -> Iterable[Path]:
    for ext in (".png", ".tif", ".tiff"):
        yield from root.rglob(f"*{ext}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild handwritten Baybayin .box files per codepoint.")
    parser.add_argument("--root", dest="roots", action="append", required=True, help="Directory containing images.")
    parser.add_argument("--threshold", type=int, default=220, help="Binary threshold (0-255).")
    parser.add_argument("--margin", type=int, default=2, help="Pixels to expand each bounding box.")
    parser.add_argument("--review-dir", type=Path, default=Path("handwritten_rebox_review"), help="Directory to collect files needing manual review.")
    args = parser.parse_args()

    review_dir = args.review_dir
    review_dir.mkdir(parents=True, exist_ok=True)

    total = ok = 0
    issues = []

    for root in args.roots:
        root_path = Path(root)
        print(f"Scanning {root_path}")
        for image_path in iter_images(root_path):
            total += 1
            success, reason = process_image(image_path, args.threshold, args.margin)
            if success:
                ok += 1
                continue

            issues.append((image_path, reason))
            rel = image_path.relative_to(root_path)
            target_dir = review_dir / rel.parent
            target_dir.mkdir(parents=True, exist_ok=True)
            for suffix in (".png", ".tif", ".tiff", ".gt.txt", ".box"):
                candidate = image_path.with_suffix(suffix)
                if candidate.exists():
                    target = target_dir / candidate.name
                    if target.exists():
                        target.unlink()
                    candidate.replace(target)

    print(f"Rebuilt {ok}/{total} box files.")
    if issues:
        log_path = review_dir / "problems.txt"
        with log_path.open("w", encoding="utf-8") as fh:
            for path, reason in issues:
                fh.write(f"{path}\t{reason}\n")
        print(f"Flagged {len(issues)} files for review (see {log_path}).")
    else:
        print("No issues encountered.")


if __name__ == "__main__":
    main()
