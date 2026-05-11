#!/usr/bin/env python3
"""Regenerate Doctrina syllables with pamudpod by omitting explicit '+' input."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from render_baybayin_namin import BaybayinTextRenderer, segment_baybayin  # type: ignore
from generate_boxes_and_lstmf import write_box_file  # type: ignore
from tools.render_bagwis_characters import (  # type: ignore
    MARK_TOKENS,
    SPECIAL_INPUT,
    VOWEL_INPUT,
    MARK_INPUT,
    build_base_input,
    boxes_for_tokens,
    components_to_entries,
    extract_components,
    split_single_box,
)


def encode_without_pamudpod(
    tokens: List[str],
    base_input: Dict[str, str],
    auto_inherent_vowel: bool,
) -> Tuple[str, Dict[int, int]]:
    parts: List[str] = []
    cluster_map: Dict[int, int] = {}
    offset = 0
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in VOWEL_INPUT:
            seq = VOWEL_INPUT[token]
            parts.append(seq)
            for local in range(len(seq)):
                cluster_map[offset + local] = i
            offset += len(seq)
            i += 1
            continue

        if token in base_input:
            seq = base_input[token]
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

        if token in MARK_INPUT and token != "᜔":
            seq = MARK_INPUT[token]
            parts.append(seq)
            for local in range(len(seq)):
                cluster_map[offset + local] = i
            offset += len(seq)
            i += 1
            continue

        if token == "᜔":
            # Skip explicit '+' input; virama is attached to the consonant glyph.
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


def derive_entries(
    image,
    hb_boxes,
    cluster_map: Dict[int, int],
    tokens: List[str],
) -> List:
    from tools.render_bagwis_characters import BoxEntry  # type: ignore

    try:
        entries = boxes_for_tokens(hb_boxes, cluster_map, tokens, image.size)
    except ValueError:
        try:
            comps = extract_components(image)
            entries = components_to_entries(comps, tokens, image.size)
        except ValueError:
            if len(hb_boxes) == 1:
                split_boxes = split_single_box(
                    (
                        int(hb_boxes[0][0]),
                        int(hb_boxes[0][1]),
                        int(hb_boxes[0][2]),
                        int(hb_boxes[0][3]),
                    ),
                    tokens,
                )
                entries = components_to_entries(split_boxes, tokens, image.size)
            else:
                raise
    for entry, token in zip(entries, tokens):
        entry.char = token
    return entries


def regenerate_samples(
    root: Path,
    renderer: BaybayinTextRenderer,
    base_input: Dict[str, str],
    auto_inherent_vowel: bool,
) -> int:
    regenerated = 0
    for gt_path in sorted(root.glob("*.gt.txt")):
        text = gt_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        tokens = segment_baybayin(text, group_marks=False)
        if "᜔" not in tokens:
            continue
        try:
            input_text, cluster_map = encode_without_pamudpod(tokens, base_input, auto_inherent_vowel)
        except ValueError as exc:
            print(f"Skipping {gt_path.name}: {exc}")
            continue
        image, hb_boxes = renderer.render_with_boxes(input_text)
        try:
            entries = derive_entries(image, hb_boxes, cluster_map, tokens)
        except ValueError as exc:
            print(f"Failed {gt_path.name}: {exc}")
            continue
        stem = gt_path.with_suffix("")
        image.save(stem.with_suffix(".tif"), format="TIFF", compression="tiff_deflate")
        write_box_file(stem.with_suffix(".box"), entries)
        regenerated += 1
    return regenerated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate Doctrina syllables with pamudpod using implicit virama glyphs."
    )
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Path to the Doctrina character directory to update.",
    )
    parser.add_argument(
        "--font",
        type=Path,
        required=True,
        help="Path to the font to use for rendering.",
    )
    parser.add_argument("--font-size", type=int, default=128)
    parser.add_argument("--margin", type=int, default=48)
    parser.add_argument(
        "--nga-sequence",
        type=str,
        default="N",
        help="Keystroke sequence that renders the NGA base.",
    )
    parser.add_argument(
        "--disable-auto-a",
        action="store_true",
        help="Disable automatic insertion of inherent 'a' after consonants.",
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
    auto_inherent = not args.disable_auto_a
    regenerated = regenerate_samples(args.root.resolve(), renderer, base_input, auto_inherent)
    print(f"Regenerated {regenerated} pamudpod samples in {args.root}.")


if __name__ == "__main__":
    main()
