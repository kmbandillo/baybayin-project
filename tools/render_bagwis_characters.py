#!/usr/bin/env python3
"""Render Bagwis Baybayin characters (single syllable per image)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from render_baybayin_namin import BaybayinTextRenderer, segment_baybayin  # type: ignore
from generate_boxes_and_lstmf import BoxEntry, write_box_file  # type: ignore

VOWEL_INPUT = {
    "ᜀ": "A",
    "ᜁ": "E",
    "ᜂ": "O",
}

BASE_ORDER = [
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
]

BASE_INPUT_TEMPLATE = {
    "ᜊ": "b",
    "ᜃ": "k",
    "ᜄ": "g",
    "ᜅ": "N",  # default (shift-n) for nga
    "ᜆ": "t",
    "ᜇ": "d",
    "ᜈ": "n",
    "ᜉ": "p",
    "ᜋ": "m",
    "ᜌ": "y",
    "ᜎ": "l",
    "ᜏ": "w",
    "ᜐ": "s",
    "ᜑ": "h",
}

MARK_INPUT = {
    "": "",
    "ᜒ": "e",
    "ᜓ": "o",
    "᜔": "+",
}

MARK_TOKENS = {"ᜒ", "ᜓ", "᜔"}

SPECIAL_INPUT = {
    "᜶": ".",
    "᜵": ",",
}


def build_base_input(nga_sequence: str) -> Dict[str, str]:
    base_input = dict(BASE_INPUT_TEMPLATE)
    base_input["ᜅ"] = nga_sequence
    return base_input


def extract_components(
    image: Image.Image,
    threshold: int = 250,
    margin: int = 1,
) -> List[Tuple[int, int, int, int]]:
    """Return bounding boxes for each connected component in the glyph image."""

    mask = np.asarray(image.convert("L")) < threshold
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: List[Tuple[int, int, int, int]] = []
    neighbors = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]

    for y in range(height):
        for x in range(width):
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
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
                        min_x = min(min_x, nx)
                        max_x = max(max_x, nx)
                        min_y = min(min_y, ny)
                        max_y = max(max_y, ny)

            left = max(0, min_x - margin)
            right = min(width, max_x + 1 + margin)
            top = max(0, min_y - margin)
            bottom = min(height, max_y + 1 + margin)
            components.append((left, top, right, bottom))

    components.sort(key=lambda box: (box[0], box[1]))
    return components


def components_to_entries(
    components: List[Tuple[int, int, int, int]],
    tokens: List[str],
    image_size: Tuple[int, int],
) -> List[BoxEntry]:
    width, height = image_size
    if len(components) != len(tokens):
        raise ValueError("component/token mismatch")

    entries: List[BoxEntry] = []
    for token, (left, top, right, bottom) in zip(tokens, components):
        bottom_pix = max(0, height - bottom)
        top_pix = min(height, height - top)
        entries.append(BoxEntry(token, left, bottom_pix, right, top_pix))
    return entries


def build_label_pairs(base_input: Dict[str, str]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for glyph in VOWEL_INPUT.keys():
        pairs.append((glyph, glyph))
    for base in BASE_ORDER:
        seq = base_input[base]
        for mark in MARK_INPUT.keys():
            render_label = base + mark
            gt_base = "ᜇ" if base == "ᜍ" else base
            gt_label = gt_base + mark
            if (gt_label, render_label) not in pairs:
                pairs.append((gt_label, render_label))
    for label in SPECIAL_INPUT.keys():
        pairs.append((label, label))
    return pairs


def encode_input(
    tokens: List[str],
    base_input: Dict[str, str],
    auto_inherent_vowel: bool,
) -> Tuple[str, Dict[int, int]]:
    """Convert Baybayin tokens into the keystrokes required by the font."""

    parts: List[str] = []
    cluster_map: Dict[int, int] = {}
    offset = 0
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in VOWEL_INPUT:
            seq = VOWEL_INPUT[token]
            if seq:
                parts.append(seq)
                for local in range(len(seq)):
                    cluster_map[offset + local] = i
                offset += len(seq)
            i += 1
            continue

        if token in base_input:
            seq = base_input[token]
            if seq:
                parts.append(seq)
                for local in range(len(seq)):
                    cluster_map[offset + local] = i
                offset += len(seq)

            next_token = tokens[i + 1] if i + 1 < len(tokens) else None
            if auto_inherent_vowel and next_token not in MARK_TOKENS:
                parts.append("a")
                cluster_map[offset] = i
                offset += 1
            i += 1
            continue

        if token in MARK_INPUT and token:
            seq = MARK_INPUT[token]
            if seq:
                parts.append(seq)
                for local in range(len(seq)):
                    cluster_map[offset + local] = i
                offset += len(seq)
            i += 1
            continue

        if token in SPECIAL_INPUT:
            seq = SPECIAL_INPUT[token]
            parts.append(seq)
            for local in range(len(seq)):
                cluster_map[offset + local] = i
            offset += len(seq)
            i += 1
            continue

        raise ValueError(f"No Bagwis input for token {token!r}")

    return "".join(parts), cluster_map


def boxes_for_tokens(
    hb_boxes: List[Tuple[float, float, float, float, int]],
    cluster_map: Dict[int, int],
    tokens: List[str],
    image_size: Tuple[int, int],
) -> List[BoxEntry]:
    width, height = image_size
    accum: List[Tuple[int, int, int, int] | None] = [None] * len(tokens)
    for x0, y0, x1, y1, cluster in hb_boxes:
        idx = cluster_map.get(cluster)
        if idx is None:
            continue
        left = max(0, int(x0))
        right = min(width, int(x1))
        top = max(0, int(y0))
        bottom = min(height, int(y1))
        existing = accum[idx]
        if existing is None:
            accum[idx] = (left, top, right, bottom)
        else:
            ex_l, ex_t, ex_r, ex_b = existing
            accum[idx] = (
                min(ex_l, left),
                min(ex_t, top),
                max(ex_r, right),
                max(ex_b, bottom),
            )

    if any(bounds is None for bounds in accum) and len(hb_boxes) == 1 and len(tokens) > 1:
        split_boxes = split_single_box(
            (int(hb_boxes[0][0]), int(hb_boxes[0][1]), int(hb_boxes[0][2]), int(hb_boxes[0][3])),
            tokens,
        )
        accum = split_boxes  # type: ignore

    entries: List[BoxEntry] = []
    for token, bounds in zip(tokens, accum):
        if bounds is None:
            raise ValueError(f"No box recorded for token {token!r}")
        left, top, right, bottom = bounds
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


def apply_baseline_shift(image: Image.Image, shift: int) -> Image.Image:
    if shift == 0:
        return image
    width, height = image.size
    background = 255 if image.mode == "L" else 0
    shifted = Image.new(image.mode, (width, height), color=background)
    shifted.paste(image, (0, -shift))
    return shifted


def render_dataset(
    font_path: Path,
    output_dir: Path,
    font_size: int,
    margin: int,
    nga_sequence: str,
    auto_inherent_vowel: bool,
    direct_input: bool = False,
    baseline_shift: int = 0,
) -> None:
    renderer = BaybayinTextRenderer(
        font_path=font_path,
        font_size=font_size,
        margin=margin,
        line_spacing=1.2,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    base_input = build_base_input(nga_sequence)
    label_pairs = build_label_pairs(base_input)
    count = 0
    for idx, (gt_label, render_label) in enumerate(label_pairs, start=1):
        if gt_label in SPECIAL_INPUT:
            tokens_render = [render_label]
            tokens_gt = [gt_label]
            seq = SPECIAL_INPUT[gt_label]
            cluster_map = {i: 0 for i in range(len(seq))}
            input_text = seq
        else:
            tokens_render = segment_baybayin(render_label, group_marks=False)
            tokens_gt = segment_baybayin(gt_label, group_marks=False)
            if direct_input:
                input_text = render_label
                cluster_map = {idx: idx for idx in range(len(tokens_render))}
            else:
                try:
                    input_text, cluster_map = encode_input(
                        tokens_render, base_input, auto_inherent_vowel
                    )
                except ValueError as exc:
                    print(f"Skipping {gt_label!r}: {exc}")
                    continue

        image, hb_boxes = renderer.render_with_boxes(input_text)
        try:
            entries = boxes_for_tokens(hb_boxes, cluster_map, tokens_render, image.size)
        except ValueError:
            try:
                comps = extract_components(image)
                entries = components_to_entries(comps, tokens_render, image.size)
            except ValueError:
                try:
                    if hb_boxes:
                        left = min(int(b[0]) for b in hb_boxes)
                        top = min(int(b[1]) for b in hb_boxes)
                        right = max(int(b[2]) for b in hb_boxes)
                        bottom = max(int(b[3]) for b in hb_boxes)
                    else:
                        left = top = 0
                        right, bottom = image.size
                    split_boxes = split_single_box((left, top, right, bottom), tokens_render)
                    entries = components_to_entries(split_boxes, tokens_render, image.size)
                except ValueError:
                    print(f"Skipping {gt_label!r}: unable to derive boxes")
                    continue

        if baseline_shift:
            image = apply_baseline_shift(image, baseline_shift)
            img_height = image.size[1]
            for entry in entries:
                entry.top = max(0, min(img_height, entry.top + baseline_shift))
                entry.bottom = max(0, min(img_height, entry.bottom + baseline_shift))

        if len(entries) != len(tokens_gt):
            print(f"Skipping {gt_label!r}: mismatch tokens vs entries")
            continue
        for entry, char in zip(entries, tokens_gt):
            entry.char = char

        base = output_dir / f"bagwis_char_{idx:04d}"
        image.save(base.with_suffix(".png"))
        image.save(base.with_suffix(".tif"), format="TIFF", compression="tiff_deflate")
        write_box_file(base.with_suffix(".box"), entries)
        base.with_suffix(".gt.txt").write_text(gt_label + "\n", encoding="utf-8")
        count += 1

    print(f"Rendered {count} samples into {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Bagwis Baybayin characters.")
    parser.add_argument("--font", type=Path, default=Path("font/BagwisBaybayinFontRegular-ZV3MK.ttf"))
    parser.add_argument("--output", type=Path, default=Path("bagwis_dataset/characters"))
    parser.add_argument("--font-size", type=int, default=128)
    parser.add_argument("--margin", type=int, default=48)
    parser.add_argument(
        "--nga-sequence",
        type=str,
        default="N",
        help="Keystroke sequence that renders the NGA base (default shift-N).",
    )
    parser.add_argument(
        "--disable-auto-a",
        action="store_true",
        help="Disable automatic insertion of inherent 'a' after consonants.",
    )
    parser.add_argument(
        "--direct-input",
        action="store_true",
        help="Send Baybayin text directly to the font (skip Bagwis transliteration).",
    )
    parser.add_argument(
        "--baseline-shift",
        type=int,
        default=0,
        help="Vertical pixel offset applied to rendered glyphs (positive = move up).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_dataset(
        args.font,
        args.output,
        args.font_size,
        args.margin,
        args.nga_sequence,
        auto_inherent_vowel=not args.disable_auto_a,
        direct_input=args.direct_input,
        baseline_shift=args.baseline_shift,
    )


if __name__ == "__main__":
    main()
