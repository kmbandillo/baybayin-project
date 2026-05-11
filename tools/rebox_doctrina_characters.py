#!/usr/bin/env python3
"""Rebuild .box files for the Doctrina character datasets using component boxes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable, List, Tuple

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from render_baybayin_namin import segment_baybayin  # type: ignore
from generate_boxes_and_lstmf import BoxEntry, write_box_file  # type: ignore
from tools.render_bagwis_characters import (  # type: ignore
    MARK_TOKENS,
    components_to_entries,
    extract_components,
    split_single_box,
)


def iter_samples(root: Path) -> Iterable[Tuple[Path, Path]]:
    for gt_path in sorted(root.rglob("*.gt.txt")):
        base = gt_path.with_suffix("")
        for ext in (".tif", ".tiff", ".png"):
            image_path = base.with_suffix(ext)
            if image_path.exists():
                yield gt_path, image_path
                break


def load_tokens(gt_path: Path) -> List[str]:
    text = gt_path.read_text(encoding="utf-8")
    label = "".join(ch for ch in text if not ch.isspace())
    return segment_baybayin(label, group_marks=False)


def union_boxes(boxes: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[2] for box in boxes)
    bottom = max(box[3] for box in boxes)
    return left, top, right, bottom


def pick_mark_component(
    components: List[Tuple[int, int, int, int]],
    mark_char: str,
) -> int:
    if mark_char == "ᜒ":
        return min(range(len(components)), key=lambda i: (components[i][1], components[i][0]))
    if mark_char == "ᜓ":
        return max(range(len(components)), key=lambda i: (components[i][3], -components[i][0]))
    if mark_char == "᜔":
        return max(range(len(components)), key=lambda i: (components[i][3], components[i][2]))
    return len(components) - 1


def resolve_component_boxes(
    components: List[Tuple[int, int, int, int]],
    tokens: List[str],
) -> List[Tuple[int, int, int, int]]:
    if not components:
        raise ValueError("no components found")
    if len(components) == len(tokens):
        return components
    if len(tokens) == 1:
        return [union_boxes(components)]
    if len(tokens) == 2 and tokens[1] in MARK_TOKENS:
        mark_idx = pick_mark_component(components, tokens[1])
        mark_box = components[mark_idx]
        base_components = [box for idx, box in enumerate(components) if idx != mark_idx]
        if not base_components:
            base_components = [mark_box]
        base_box = union_boxes(base_components)
        return [base_box, mark_box]
    if len(components) == 1 and len(tokens) > 1:
        return split_single_box(components[0], tokens)

    merged = union_boxes(components)
    if len(tokens) > 1:
        return split_single_box(merged, tokens)
    return [merged]


def derive_entries(
    image_path: Path,
    tokens: List[str],
    threshold: int,
    margin: int,
) -> List[BoxEntry]:
    with Image.open(image_path) as image:
        comps = extract_components(image, threshold=threshold, margin=margin)
        boxes = resolve_component_boxes(comps, tokens)
        entries = components_to_entries(boxes, tokens, image.size)

    for entry, token in zip(entries, tokens):
        entry.char = token
    return entries


def process_dataset(root: Path, threshold: int, margin: int, dry_run: bool) -> Tuple[int, int]:
    total = success = 0
    for gt_path, image_path in iter_samples(root):
        total += 1
        tokens = load_tokens(gt_path)
        if not tokens:
            print(f"Skipping empty label: {gt_path}")
            continue
        try:
            entries = derive_entries(image_path, tokens, threshold, margin)
        except ValueError as exc:
            print(f"Failed {gt_path}: {exc}")
            continue
        success += 1
        if dry_run:
            continue
        write_box_file(image_path.with_suffix(".box"), entries)
    return success, total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebox Doctrina character datasets using the Bagwis component method."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("dataset/characters/doctrina_characters_clear_noauto"),
        help="Directory that holds rendered Doctrina characters.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=250,
        help="Grayscale threshold used to segment glyphs (0-255).",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=1,
        help="Margin in pixels added around each component box.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report counts without rewriting the .box files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    success, total = process_dataset(args.root.resolve(), args.threshold, args.margin, args.dry_run)
    suffix = " (dry run)" if args.dry_run else ""
    print(f"Reboxed {success}/{total} Doctrina samples{suffix}.")


if __name__ == "__main__":
    main()
