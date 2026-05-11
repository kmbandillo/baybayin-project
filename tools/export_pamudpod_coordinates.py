#!/usr/bin/env python3
"""Export base consonant vs pamudpod coordinates for Doctrina samples."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tools import rebox_doctrina_characters as rebox  # type: ignore

LATIN_MAP = {
    "ᜊ": "B",
    "ᜃ": "K",
    "ᜄ": "G",
    "ᜅ": "NG",
    "ᜆ": "T",
    "ᜇ": "D",
    "ᜈ": "N",
    "ᜉ": "P",
    "ᜋ": "M",
    "ᜌ": "Y",
    "ᜎ": "L",
    "ᜏ": "W",
    "ᜐ": "S",
    "ᜑ": "H",
}


def map_label(token: str) -> str:
    if token == "᜔":
        return "+"
    return LATIN_MAP.get(token, token)


def pick_indices(tokens: Sequence[str]) -> Tuple[int, int]:
    base_indices = [idx for idx, tok in enumerate(tokens) if tok != "᜔"]
    mark_indices = [idx for idx, tok in enumerate(tokens) if tok == "᜔"]
    if not base_indices or not mark_indices:
        raise ValueError("Expected one consonant and one pamudpod")
    return base_indices[0], mark_indices[0]


def export_coordinates(
    root: Path,
    output: Path,
    threshold: int,
    margin: int,
) -> Tuple[int, int]:
    rows: List[Tuple[str, str, str, int, int, int, int]] = []
    total = exported = 0

    for gt_path, image_path in rebox.iter_samples(root):
        tokens = rebox.load_tokens(gt_path)
        if not tokens or "᜔" not in tokens:
            continue
        total += 1
        try:
            entries = rebox.derive_entries(image_path, tokens, threshold, margin)
        except ValueError as exc:
            print(f"Skipping {gt_path}: {exc}")
            continue
        base_idx, mark_idx = pick_indices(tokens)
        entries_list = list(entries)
        base_entry = entries_list[base_idx]
        mark_entry = entries_list[mark_idx]
        rel_name = image_path.relative_to(root)
        rows.append(
            (
                str(rel_name),
                map_label(tokens[base_idx]),
                "main",
                base_entry.left,
                base_entry.bottom,
                base_entry.right,
                base_entry.top,
            )
        )
        rows.append(
            (
                str(rel_name),
                map_label(tokens[mark_idx]),
                "pamudpod",
                mark_entry.left,
                mark_entry.bottom,
                mark_entry.right,
                mark_entry.top,
            )
        )
        exported += 1

    if rows:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["sample", "label", "component", "left", "bottom", "right", "top"])
            writer.writerows(rows)

    return exported, total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export coordinates for Doctrina consonants with pamudpod marks."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("dataset/characters/doctrina_characters_noauto"),
        help="Directory containing Doctrina character samples.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/characters/doctrina_characters_noauto/pamudpod_coordinates.csv"),
        help="Path to the CSV file to write.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=250,
        help="Binary threshold for component extraction (0-255).",
    )
    parser.add_argument(
        "--margin",
        type=int,
        default=1,
        help="Margin in pixels to add around each component before exporting.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exported, total = export_coordinates(args.root.resolve(), args.output.resolve(), args.threshold, args.margin)
    print(f"Exported pamudpod coords for {exported}/{total} samples that contain pamudpod.")


if __name__ == "__main__":
    main()
