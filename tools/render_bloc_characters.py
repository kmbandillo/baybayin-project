#!/usr/bin/env python3
"""Render Baybayin Bloc font characters (separate glyph + diacritic boxes)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from render_baybayin_namin import BaybayinTextRenderer, segment_baybayin  # type: ignore
from generate_boxes_and_lstmf import (
    BoxEntry,
    write_box_file,
    convert_boxes_to_entries,
)  # type: ignore

VOWELS = ["ᜀ", "ᜁ", "ᜂ"]
BASES = [
    "ᜊ",
    "ᜃ",
    "ᜄ",
    "ᜅ",
    "ᜆ",
    "ᜇ",
    "ᜈ",
    "ᜉ",
    "ᜋ",
    "ᜌ",
    "ᜎ",
    "ᜏ",
    "ᜐ",
    "ᜑ",
    "ᜍ",
]
MARKS = ["", "ᜒ", "ᜓ", "᜔"]
SPECIALS = ["᜵", "᜶"]


def build_label_pairs() -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for vowel in VOWELS:
        pairs.append((vowel, vowel))
    for base in BASES:
        gt_base = "ᜇ" if base == "ᜍ" else base
        for mark in MARKS:
            pairs.append((gt_base + mark, base + mark))
    for special in SPECIALS:
        pairs.append((special, special))
    return pairs


def extract_components(image: Image.Image, threshold: int = 250, margin: int = 1) -> List[Tuple[int, int, int, int]]:
    mask = np.asarray(image.convert("L")) < threshold
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
            comps.append((left, top, right, bottom))

    comps.sort(key=lambda b: (b[0], b[1]))
    return comps


def components_to_entries(
    comps: List[Tuple[int, int, int, int]],
    tokens: List[str],
    image_size: Tuple[int, int],
) -> List[BoxEntry]:
    width, height = image_size
    if len(comps) != len(tokens):
        raise ValueError("component/token mismatch")
    entries: List[BoxEntry] = []
    for token, (left, top, right, bottom) in zip(tokens, comps):
        bottom_pix = max(0, height - bottom)
        top_pix = min(height, height - top)
        entries.append(BoxEntry(token, left, bottom_pix, right, top_pix))
    return entries


def split_single_box(box: Tuple[int, int, int, int], tokens: List[str]) -> List[Tuple[int, int, int, int]]:
    left, top, right, bottom = box
    width = right - left
    height = bottom - top
    base_box = [left, top, right, bottom]
    result: List[Tuple[int, int, int, int] | None] = [None] * len(tokens)
    for idx, token in enumerate(tokens):
        if token == "ᜒ":
            mark_h = max(1, int(height * 0.35))
            result[idx] = (left, top, right, top + mark_h)
            base_box[1] = top + mark_h
        elif token == "ᜓ":
            mark_h = max(1, int(height * 0.35))
            result[idx] = (left, bottom - mark_h, right, bottom)
            base_box[3] = bottom - mark_h
        elif token == "᜔":
            mark_w = max(1, int(width * 0.25))
            result[idx] = (right - mark_w, top, right, bottom)
            base_box[2] = right - mark_w
    for idx in range(len(tokens)):
        if result[idx] is None:
            result[idx] = tuple(base_box)
    return result  # type: ignore[arg-type]


def render_dataset(font_path: Path, output_dir: Path, font_size: int, margin: int) -> None:
    renderer = BaybayinTextRenderer(
        font_path=font_path,
        font_size=font_size,
        margin=margin,
        line_spacing=1.2,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for idx, (gt_label, render_label) in enumerate(build_label_pairs(), start=1):
        tokens_render = segment_baybayin(render_label, group_marks=False)
        tokens_gt = segment_baybayin(gt_label, group_marks=False)

        image, hb_boxes = renderer.render_with_boxes(render_label)
        box_list = [(x0, y0, x1, y1) for (x0, y0, x1, y1, _cluster) in hb_boxes]
        try:
            entries = convert_boxes_to_entries(box_list, tokens_render, image.size)
        except ValueError:
            try:
                comps = extract_components(image)
                entries = components_to_entries(comps, tokens_render, image.size)
            except ValueError:
                try:
                    if len(box_list) == 1:
                        split = split_single_box(box_list[0], tokens_render)
                        entries = convert_boxes_to_entries(split, tokens_render, image.size)
                    else:
                        raise ValueError("Unable to split glyphs")
                except ValueError as exc:
                    print(f"Skipping {gt_label!r}: {exc}")
                    continue

        if len(entries) != len(tokens_gt):
            print(f"Skipping {gt_label!r}: mismatch tokens vs entries")
            continue
        for entry, token_char in zip(entries, tokens_gt):
            entry.char = token_char

        base = output_dir / f"bloc_char_{idx:04d}"
        image.save(base.with_suffix(".png"))
        image.save(base.with_suffix(".tif"), format="TIFF", compression="tiff_deflate")
        write_box_file(base.with_suffix(".box"), entries)
        base.with_suffix(".gt.txt").write_text(gt_label + "\n", encoding="utf-8")
        count += 1

    print(f"Rendered {count} samples into {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Baybayin Bloc characters.")
    parser.add_argument("--font", type=Path, default=Path("font/Baybayin_Bloc.ttf"))
    parser.add_argument("--output", type=Path, default=Path("bloc_dataset/characters"))
    parser.add_argument("--font-size", type=int, default=128)
    parser.add_argument("--margin", type=int, default=48)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_dataset(args.font, args.output, args.font_size, args.margin)


if __name__ == "__main__":
    main()
