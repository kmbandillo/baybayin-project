#!/usr/bin/env python3
"""
Rebox handwritten Baybayin word images so that each Baybayin codepoint (base glyph and
its kudlit/pamudpod marks) receives an individual bounding box. Ambiguous samples are
copied into a review folder for manual inspection.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

MARK_TOP = {"ᜒ"}
MARK_BOTTOM = {"ᜓ", "᜔"}
MARKS = MARK_TOP.union(MARK_BOTTOM)


def load_binary(image_path: Path, threshold: int) -> np.ndarray:
    img = Image.open(image_path).convert("L")
    arr = np.array(img)
    return arr < threshold


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
            pixels = [(y, x)]

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
                        pixels.append((ny, nx))

            comps.append(
                {
                    "left": min_x,
                    "top": min_y,
                    "right": max_x + 1,
                    "bottom": max_y + 1,
                    "area": count,
                    "cx": sum_x / count,
                    "cy": sum_y / count,
                    "pixels": pixels,
                }
            )

    comps.sort(key=lambda c: (c["left"], c["top"]))
    return comps


def group_tokens(text: str) -> Optional[List[dict]]:
    tokens = [ch for ch in text if ch not in {" ", "\n", "\r", "\t"}]
    groups: List[dict] = []
    for ch in tokens:
        if ch in MARKS:
            if not groups:
                return None
            groups[-1]["marks"].append(ch)
        else:
            groups.append({"base": ch, "marks": []})
    return groups


def column_boundaries(mask: np.ndarray, count: int) -> List[int]:
    width = mask.shape[1]
    col_sum = mask.sum(axis=0)
    boundaries = [0]
    half_window = max(2, width // (count * 4))
    last = 0
    for i in range(1, count):
        target = int(round(i * width / count))
        start = max(last + 1, target - half_window)
        end = min(width - 1, target + half_window)
        best = start
        best_val = col_sum[start]
        for x in range(start, end + 1):
            val = col_sum[x]
            if val < best_val:
                best_val = val
                best = x
        boundaries.append(max(best, last + 1))
        last = boundaries[-1]
    boundaries.append(width)
    return boundaries


def bounding_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    rows, cols = np.where(mask)
    if rows.size == 0 or cols.size == 0:
        return None
    return cols.min(), rows.min(), cols.max() + 1, rows.max() + 1


def split_region(mask: np.ndarray, mark_char: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    rows = mask.sum(axis=1)
    total = int(rows.sum())
    if total == 0:
        return None
    height = mask.shape[0]
    if mark_char in MARK_TOP:
        cumulative = np.cumsum(rows)
        target = max(1, int(total * 0.2))
        cut = int(np.searchsorted(cumulative, target))
        cut = max(1, min(cut, height - 1))
        mark_mask = mask.copy()
        mark_mask[cut:, :] = False
        base_mask = mask.copy()
        base_mask[:cut, :] = False
    else:
        cumulative = np.cumsum(rows[::-1])
        target = max(1, int(total * 0.2))
        cut = int(np.searchsorted(cumulative, target))
        cut = max(1, min(cut, height - 1))
        boundary = height - cut
        mark_mask = mask.copy()
        mark_mask[:boundary, :] = False
        base_mask = mask.copy()
        base_mask[boundary:, :] = False
    return base_mask, mark_mask


def expand_box(box: Tuple[int, int, int, int], width: int, height: int, margin: int) -> Tuple[int, int, int, int]:
    left, top, right, bottom = box
    return (
        max(0, left - margin),
        max(0, top - margin),
        min(width, right + margin),
        min(height, bottom + margin),
    )


def iter_images(root: Path) -> Iterable[Path]:
    for ext in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
        yield from root.rglob(f"*{ext}")


def write_box(path: Path, tokens: Sequence[str], boxes: Sequence[Tuple[int, int, int, int]], height: int) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for ch, (left, top, right, bottom) in zip(tokens, boxes):
            bottom_inv = height - bottom
            top_inv = height - top
            fh.write(f"{ch} {left} {bottom_inv} {right} {top_inv} 0\n")


def choose_component(
    comps: List[dict],
    condition,
) -> Optional[int]:
    candidates = [idx for idx, comp in enumerate(comps) if condition(comp)]
    if not candidates:
        return None
    candidates.sort(key=lambda i: (-comps[i]["area"], comps[i]["top"]))
    return candidates[0]


def process_word(image_path: Path, threshold: int, margin: int) -> Tuple[bool, List[str]]:
    gt_path = image_path.with_suffix(".gt.txt")
    if not gt_path.exists():
        return False, ["missing_gt"]

    text = gt_path.read_text(encoding="utf-8")
    groups = group_tokens(text)
    if not groups:
        return False, ["token_grouping_failed"]

    tokens = [ch for ch in text if ch not in {" ", "\n", "\r", "\t"}]
    mask = load_binary(image_path, threshold)
    height, width = mask.shape
    boundaries = column_boundaries(mask, len(groups))

    boxes: List[Tuple[int, int, int, int]] = []
    issues: List[str] = []
    token_idx = 0

    for group_idx, group in enumerate(groups):
        seg_left = max(0, boundaries[group_idx] - 2)
        seg_right = min(width, boundaries[group_idx + 1] + 2)
        segment = mask[:, seg_left:seg_right]
        rows = np.where(segment.any(axis=1))[0]
        if rows.size == 0:
            issues.append(f"empty_segment:{group_idx}")
            return False, issues
        top = rows.min()
        bottom = rows.max() + 1
        segment_crop = segment[top:bottom, :].copy()
        comps = connected_components(segment_crop)
        if not comps:
            issues.append(f"no_components:{group_idx}")
            return False, issues

        base_idx = max(range(len(comps)), key=lambda i: comps[i]["area"])
        base_comp = comps[base_idx]
        base_mask = np.zeros_like(segment_crop, dtype=bool)
        for py, px in base_comp["pixels"]:
            base_mask[py, px] = True

        remaining_mask = segment_crop.copy()
        remaining = [comp for i, comp in enumerate(comps) if i != base_idx]
        mark_boxes_rel: List[Tuple[int, int, int, int]] = []

        base_height = base_comp["bottom"] - base_comp["top"]
        base_top = base_comp["top"]
        base_bottom = base_comp["bottom"]

        for mark_char in group["marks"]:
            mark_bbox_rel: Optional[Tuple[int, int, int, int]] = None
            if remaining:
                if mark_char in MARK_TOP:
                    mark_idx = choose_component(
                        remaining,
                        lambda c: c["cy"] <= base_top + max(2, 0.35 * base_height),
                    )
                else:
                    mark_idx = choose_component(
                        remaining,
                        lambda c: c["cy"] >= base_bottom - max(2, 0.35 * base_height),
                    )
            else:
                mark_idx = None

            if mark_idx is None:
                split_result = split_region(remaining_mask, mark_char)
                if split_result is None:
                    issues.append(f"split_failed:{group_idx}")
                    return False, issues
                base_mask_split, mark_mask = split_result
                mark_bbox_rel = bounding_from_mask(mark_mask)
                if mark_bbox_rel is None:
                    issues.append(f"mark_empty:{group_idx}")
                    return False, issues
                remaining_mask = base_mask_split
                base_mask = base_mask_split
            else:
                comp = remaining.pop(mark_idx)
                mark_bbox_rel = (
                    comp["left"],
                    comp["top"],
                    comp["right"],
                    comp["bottom"],
                )
                for py, px in comp["pixels"]:
                    remaining_mask[py, px] = False

            mark_boxes_rel.append(mark_bbox_rel)

        base_bbox_rel = bounding_from_mask(base_mask)
        if base_bbox_rel is None:
            base_bbox_rel = bounding_from_mask(remaining_mask)
        if base_bbox_rel is None:
            issues.append(f"base_empty:{group_idx}")
            return False, issues

        # Convert relative boxes to absolute coordinates and append (base first, then marks).
        base_abs = (
            seg_left + base_bbox_rel[0],
            top + base_bbox_rel[1],
            seg_left + base_bbox_rel[2],
            top + base_bbox_rel[3],
        )
        boxes.append(expand_box(base_abs, width, height, margin))
        token_idx += 1

        for mark_bbox_rel, mark_char in zip(mark_boxes_rel, group["marks"]):
            mark_abs = (
                seg_left + mark_bbox_rel[0],
                top + mark_bbox_rel[1],
                seg_left + mark_bbox_rel[2],
                top + mark_bbox_rel[3],
            )
            boxes.append(expand_box(mark_abs, width, height, max(1, margin // 2)))
            token_idx += 1

    if len(boxes) != len(tokens):
        return False, ["token_box_mismatch"]

    write_box(image_path.with_suffix(".box"), tokens, boxes, height)
    return True, issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebox handwritten Baybayin word dataset.")
    parser.add_argument("--root", type=Path, required=True, help="Word dataset directory.")
    parser.add_argument("--threshold", type=int, default=220, help="Binary threshold (0-255).")
    parser.add_argument("--margin", type=int, default=2, help="Extra pixels to expand each box.")
    parser.add_argument("--review-dir", type=Path, default=Path("word_rebox_review"), help="Directory for ambiguous samples.")
    args = parser.parse_args()

    root = args.root.resolve()
    review_dir = args.review_dir.resolve()
    review_dir.mkdir(parents=True, exist_ok=True)

    total = ok = 0
    problems: List[str] = []

    for image_path in iter_images(root):
        success, notes = process_word(image_path, args.threshold, args.margin)
        total += 1
        if success:
            ok += 1
        else:
            rel = image_path.relative_to(root)
            target_dir = review_dir / rel.parent
            target_dir.mkdir(parents=True, exist_ok=True)
            for suffix in (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".gt.txt", ".box"):
                candidate = image_path.with_suffix(suffix)
                if candidate.exists():
                    dest = target_dir / candidate.name
                    dest.write_bytes(candidate.read_bytes())
            problems.append(f"{rel}\t{'|'.join(notes)}")

    print(f"Reboxed {ok}/{total} word samples.")
    if problems:
        (review_dir / "problems.txt").write_text("\n".join(problems) + "\n", encoding="utf-8")
        print(f"Flagged {len(problems)} ambiguous samples in {review_dir}.")
    else:
        print("No ambiguous samples encountered.")


if __name__ == "__main__":
    main()
