#!/usr/bin/env python3
"""Regenerate Doctrina base syllables (e.g., ᜊ) using explicit 'consonant + a' input."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from render_baybayin_namin import BaybayinTextRenderer, segment_baybayin  # type: ignore
from generate_boxes_and_lstmf import write_box_file  # type: ignore
from tools.render_bagwis_characters import (  # type: ignore
    BASE_ORDER,
    build_base_input,
    encode_input,
    boxes_for_tokens,
)


BASE_SET = set(BASE_ORDER)


def iter_base_gt(root: Path) -> Iterable[Path]:
    for gt_path in sorted(root.glob("*.gt.txt")):
        label = gt_path.read_text(encoding="utf-8").strip()
        if label in BASE_SET:
            yield gt_path


def build_text_with_a(
    tokens: List[str],
    base_input: dict[str, str],
) -> Tuple[str, dict[int, int]]:
    text, cluster_map = encode_input(tokens, base_input, auto_inherent_vowel=False)
    if len(tokens) == 1 and tokens[0] in BASE_SET:
        offset = len(text)
        text += "a"
        cluster_map[offset] = 0
    return text, cluster_map


def redraw_sample(
    gt_path: Path,
    renderer: BaybayinTextRenderer,
    base_input: dict[str, str],
) -> None:
    label = gt_path.read_text(encoding="utf-8").strip()
    tokens = segment_baybayin(label, group_marks=False)
    text, cluster_map = build_text_with_a(tokens, base_input)
    image, hb_boxes = renderer.render_with_boxes(text)
    entries = boxes_for_tokens(hb_boxes, cluster_map, tokens, image.size)
    for entry, token in zip(entries, tokens):
        entry.char = token
    stem = gt_path.with_suffix("")
    image.save(stem.with_suffix(".tif"), format="TIFF", compression="tiff_deflate")
    write_box_file(stem.with_suffix(".box"), entries)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ensure Doctrina base syllables are rendered via consonant + 'a'."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("dataset/characters/doctrina_characters_noauto"),
        help="Doctrina character directory to patch.",
    )
    parser.add_argument(
        "--font",
        type=Path,
        default=Path("font/Baybayin Doctrina.otf"),
        help="Path to the Doctrina font.",
    )
    parser.add_argument("--font-size", type=int, default=128)
    parser.add_argument("--margin", type=int, default=48)
    parser.add_argument(
        "--nga-sequence",
        type=str,
        default="N",
        help="Keystroke sequence that renders the NGA base.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    renderer = BaybayinTextRenderer(
        font_path=args.font,
        font_size=args.font_size,
        margin=args.margin,
        line_spacing=1.2,
    )
    base_input = build_base_input(args.nga_sequence)
    targets = list(iter_base_gt(args.root.resolve()))
    for gt_path in targets:
        redraw_sample(gt_path, renderer, base_input)
    print(f"Regenerated {len(targets)} base syllable samples via consonant + 'a'.")


if __name__ == "__main__":
    main()
