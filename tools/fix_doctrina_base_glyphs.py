#!/usr/bin/env python3
"""Fix Doctrina consonants with inherent 'a' by redrawing them without pamudpod."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List

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


def find_base_samples(root: Path) -> List[Path]:
    targets: List[Path] = []
    for gt_path in sorted(root.glob("*.gt.txt")):
        text = gt_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        tokens = segment_baybayin(text, group_marks=False)
        if len(tokens) == 1 and tokens[0] in BASE_ORDER:
            targets.append(gt_path)
    return targets


def redraw_samples(
    targets: List[Path],
    renderer: BaybayinTextRenderer,
    base_input: dict[str, str],
) -> int:
    updated = 0
    for gt_path in targets:
        label = gt_path.read_text(encoding="utf-8").strip()
        tokens = segment_baybayin(label, group_marks=False)
        input_text, cluster_map = encode_input(tokens, base_input, auto_inherent_vowel=True)
        image, hb_boxes = renderer.render_with_boxes(input_text)
        entries = boxes_for_tokens(hb_boxes, cluster_map, tokens, image.size)
        for entry, char in zip(entries, tokens):
            entry.char = char
        stem = gt_path.with_suffix("")
        image.save(stem.with_suffix(".png"))
        image.save(stem.with_suffix(".tif"), format="TIFF", compression="tiff_deflate")
        write_box_file(stem.with_suffix(".box"), entries)
        updated += 1
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update Doctrina base consonants to remove pamudpod.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("dataset/characters/doctrina_characters_noauto"),
        help="Directory with Doctrina characters.",
    )
    parser.add_argument(
        "--font",
        type=Path,
        default=Path("font/Baybayin Doctrina.otf"),
        help="Path to the Doctrina font file.",
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
    targets = find_base_samples(args.root.resolve())
    updated = redraw_samples(targets, renderer, base_input)
    print(f"Redrew {updated} base consonant samples with inherent 'a'.")


if __name__ == "__main__":
    main()
