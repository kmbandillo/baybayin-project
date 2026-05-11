#!/usr/bin/env python3
"""
Rebuild Baybayin handwritten character .box files, splitting kudlit/pamudpod marks into
separate bounding boxes when present. Any samples we cannot confidently split are copied
into a review directory for manual inspection.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image

MARK_TOP = {"ᜒ"}
MARK_BOTTOM = {"ᜓ", "᜔"}


def load_binary(image_path: Path, threshold: int) -> np.ndarray:
    img = Image.open(image_path).convert("L")
    arr = np.array(img)
    mask = arr < threshold
    return mask


def connected_components(mask: np.ndarray) -> List[dict]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps: List[dict] = []
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            min_x = max_x = x
            min_y = max_y = y
            sum_x = x
            sum_y = y
            count = 1

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
                        sum_x += nx
                        sum_y += ny
                        count += 1

            comps.append(
                {
                    "left": min_x,
                    "top": min_y,
                    "right": max_x + 1,
                    "bottom": max_y + 1,
                    "area": count,
                    "cy": sum_y / count,
                    "cx": sum_x / count,
                }
            )

    comps.sort(key=lambda c: (c["left"], c["top"]))
    return comps


def union_boxes(comps: List[dict]) -> Tuple[int, int, int, int]:
    left = min(c["left"] for c in comps)
    top = min(c["top"] for c in comps)
    right = max(c["right"] for c in comps)
    bottom = max(c["bottom"] for c in comps)
    return left, top, right, bottom


def expand_box(box: Tuple[int, int, int, int], width: int, height: int, margin: int) -> Tuple[int, int, int, int]:
    left, top, right, bottom = box
    left = max(0, left - margin)
    top = max(0, top - margin)
    right = min(width, right + margin)
    bottom = min(height, bottom + margin)
    return left, top, right, bottom


def load_tokens(gt_path: Path) -> List[str]:
    text = gt_path.read_text(encoding="utf-8")
    return [ch for ch in text if ch not in {" ", "\n", "\r", "\t"}]


def split_by_projection(mask: np.ndarray, bbox: Tuple[int, int, int, int], mark_char: str) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    left, top, right, bottom = bbox
    sub = mask[top:bottom, left:right]
    height = bottom - top
    if height <= 1 or sub.sum() == 0:
        return bbox, bbox

    if mark_char in MARK_TOP:
        target = max(1, int(sub.sum() * 0.2))
        cumulative = 0
        cut = top + max(1, int(height * 0.25))
        for py in range(sub.shape[0]):
            row_sum = int(sub[py].sum())
            cumulative += row_sum
            if cumulative >= target:
                cut = top + py
                break
        mark_box = (left, top, right, min(bottom, cut + 1))
        base_box = (left, min(bottom, cut + 1), right, bottom)
    else:  # bottom marks
        target = max(1, int(sub.sum() * 0.2))
        cumulative = 0
        cut = bottom - max(1, int(height * 0.25))
        for py in range(sub.shape[0] - 1, -1, -1):
            row_sum = int(sub[py].sum())
            cumulative += row_sum
            if cumulative >= target:
                cut = top + py
                break
        mark_box = (left, max(top, cut), right, bottom)
        base_box = (left, top, max(left, cut), bottom)

    if base_box[3] <= base_box[1]:
        base_box = bbox
    if mark_box[3] <= mark_box[1]:
        mark_box = bbox
    return base_box, mark_box


def assign_boxes(tokens: List[str], comps: List[dict], mask: np.ndarray, margin: int) -> Tuple[List[Tuple[int, int, int, int]], bool]:
    height, width = mask.shape
    boxes: List[Tuple[int, int, int, int]] = []
    review_needed = False

    if not tokens:
        return boxes, True

    if len(tokens) == 1:
        if comps:
            base_box = union_boxes(comps)
        else:
            rows, cols = np.where(mask)
            if rows.size == 0:
                review_needed = True
                base_box = (0, 0, width, height)
            else:
                base_box = (int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1)
        boxes.append(expand_box(base_box, width, height, margin))
        return boxes, review_needed

    if len(tokens) == 2 and tokens[1] in MARK_TOP.union(MARK_BOTTOM):
        mark_char = tokens[1]
        mark_box = None
        base_components = comps.copy()
        if len(comps) >= 2:
            if mark_char in MARK_TOP:
                mark_comp = min(comps, key=lambda c: (c["cy"], c["area"]))
            else:
                mark_comp = max(comps, key=lambda c: (c["cy"], -c["area"]))
            mark_box = (mark_comp["left"], mark_comp["top"], mark_comp["right"], mark_comp["bottom"])
            base_components = [c for c in comps if c is not mark_comp]
            if not base_components:
                base_components = [mark_comp]
        if mark_box is None or len(base_components) == 0:
            rows, cols = np.where(mask)
            if rows.size == 0:
                review_needed = True
                return [], True
            bbox = (int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1)
            base_box, mark_box = split_by_projection(mask, bbox, mark_char)
        else:
            base_box = union_boxes(base_components)

        boxes.append(expand_box(base_box, width, height, margin))
        boxes.append(expand_box(mark_box, width, height, margin))
        return boxes, review_needed

    # Fallback: naive mapping left-to-right
    if len(comps) >= len(tokens):
        comps_sorted = sorted(comps, key=lambda c: (c["left"], c["top"]))
        for idx in range(len(tokens)):
            comp = comps_sorted[idx]
            box = (comp["left"], comp["top"], comp["right"], comp["bottom"])
            boxes.append(expand_box(box, width, height, margin))
        if len(comps) != len(tokens):
            review_needed = True
        return boxes, review_needed

    review_needed = True
    return [], review_needed


def write_box_file(path: Path, tokens: List[str], boxes: List[Tuple[int, int, int, int]], height: int) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for ch, (left, top, right, bottom) in zip(tokens, boxes):
            bottom_inv = height - bottom
            top_inv = height - top
            fh.write(f"{ch} {left} {bottom_inv} {right} {top_inv} 0\n")


def iter_images(root: Path) -> Iterable[Path]:
    for ext in (".tif", ".tiff", ".png"):
        yield from root.rglob(f"*{ext}")


def process(root: Path, threshold: int, margin: int, review_dir: Path) -> Tuple[int, int]:
    total = ok = 0
    for image_path in iter_images(root):
        gt_path = image_path.with_suffix(".gt.txt")
        if not gt_path.exists():
            continue
        tokens = load_tokens(gt_path)
        mask = load_binary(image_path, threshold)
        comps = connected_components(mask)
        boxes, needs_review = assign_boxes(tokens, comps, mask, margin)
        total += 1
        if needs_review or not boxes or len(boxes) != len(tokens):
            review_dir.mkdir(parents=True, exist_ok=True)
            rel = image_path.relative_to(root)
            target_dir = review_dir / rel.parent
            target_dir.mkdir(parents=True, exist_ok=True)
            for suffix in (".tif", ".tiff", ".png", ".gt.txt", ".box"):
                src = image_path.with_suffix(suffix)
                if src.exists():
                    dst = target_dir / src.name
                    dst.write_bytes(src.read_bytes())
            continue
        ok += 1
        height = mask.shape[0]
        box_path = image_path.with_suffix(".box")
        write_box_file(box_path, tokens, boxes, height)
    return ok, total


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebox Baybayin handwritten characters.")
    parser.add_argument("--root", type=Path, required=True, help="Root directory of character dataset.")
    parser.add_argument("--threshold", type=int, default=220, help="Binary threshold (0-255).")
    parser.add_argument("--margin", type=int, default=2, help="Margin in pixels to expand boxes.")
    parser.add_argument("--review-dir", type=Path, default=Path("character_rebox_review"), help="Directory to hold ambiguous samples.")
    args = parser.parse_args()

    ok, total = process(args.root.resolve(), args.threshold, args.margin, args.review_dir.resolve())
    print(f"Reboxed {ok}/{total} samples. Ambiguous cases copied to {args.review_dir}.")


if __name__ == "__main__":
    main()
