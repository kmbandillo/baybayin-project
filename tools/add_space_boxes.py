#!/usr/bin/env python3
"""Insert space box entries into phrase datasets using GT text and existing glyph boxes."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from PIL import Image


MARKS = {"ᜒ", "ᜓ", "᜔"}


@dataclass
class BoxLine:
    char: str
    left: int
    bottom: int
    right: int
    top: int
    page: int

    def clone(self) -> "BoxLine":
        return BoxLine(self.char, self.left, self.bottom, self.right, self.top, self.page)

    def formatted(self) -> str:
        return f"{self.char} {self.left} {self.bottom} {self.right} {self.top} {self.page}"


def parse_box_line(line: str) -> Optional[BoxLine]:
    stripped = line.strip()
    if not stripped:
        return None
    char = stripped[0]
    rest = stripped[1:]
    if not char.isspace():
        idx = 1
        while idx < len(stripped) and not stripped[idx].isspace():
            idx += 1
        char = stripped[:idx]
        rest = stripped[idx:]
    tokens = rest.split()
    if len(tokens) < 5:
        return None
    try:
        left, bottom, right, top, page = map(int, tokens[:5])
    except ValueError:
        return None
    return BoxLine(char, left, bottom, right, top, page)


def load_box_entries(path: Path) -> List[BoxLine]:
    entries: List[BoxLine] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        entry = parse_box_line(raw_line)
        if entry is not None:
            entries.append(entry)
    return entries


def save_box_entries(path: Path, entries: Iterable[BoxLine]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(entry.formatted() + "\n")


def find_previous_base(entries: List[BoxLine]) -> Optional[BoxLine]:
    for entry in reversed(entries):
        if entry.char == " " or entry.char == "\\n":
            continue
        if entry.char in MARKS:
            continue
        if not entry.char.strip():
            continue
        return entry
    return None


def find_next_base(entries: List[BoxLine], start_index: int) -> Optional[BoxLine]:
    for idx in range(start_index, len(entries)):
        entry = entries[idx]
        if entry.char == " " or entry.char == "\\n":
            continue
        if entry.char in MARKS:
            continue
        if not entry.char.strip():
            continue
        return entry
    return None


def insert_spaces(entries: List[BoxLine], text: str, image_height: int) -> Optional[List[BoxLine]]:
    if not text:
        return None

    base_order = [
        entry
        for entry in entries
        if entry.char not in MARKS and entry.char.strip() and entry.char != "\\n"
    ]
    if not base_order:
        return None
    base_order.sort(key=lambda e: (e.left, e.right))

    base_idx = 0
    insertions: List[tuple[Optional[BoxLine], BoxLine]] = []
    for ch in text:
        if ch == "\n":
            continue
        if ch == " ":
            prev_base = base_order[base_idx - 1] if base_idx > 0 else None
            next_base = base_order[base_idx] if base_idx < len(base_order) else None
            left = prev_base.right if prev_base is not None else 0
            if next_base is not None:
                right = next_base.left
            else:
                right = left + max(1, prev_base.width if prev_base else 1)
            if right <= left:
                right = left + 1
            insertions.append((prev_base, BoxLine(" ", left, 0, right, image_height, 0)))
            continue
        if ch not in MARKS:
            base_idx += 1

    if not insertions:
        return None

    final_entries = list(entries)
    for prev_base, space_entry in insertions:
        if prev_base is None:
            insert_idx = 0
        else:
            insert_idx = None
            for idx, entry in enumerate(final_entries):
                if entry is prev_base:
                    insert_idx = idx + 1
                    break
            if insert_idx is None:
                insert_idx = len(final_entries)
            else:
                while insert_idx < len(final_entries) and final_entries[insert_idx].char in MARKS:
                    insert_idx += 1
        final_entries.insert(insert_idx, space_entry)
    return final_entries


def find_image_path(box_path: Path) -> Optional[Path]:
    candidates = [
        box_path.with_suffix(".png"),
        box_path.with_suffix(".tif"),
        box_path.with_suffix(".tiff"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def process_directory(root: Path) -> int:
    updated = 0
    for box_path in sorted(root.glob("*.box")):
        gt_path = box_path.with_suffix(".gt.txt")
        if not gt_path.exists():
            continue
        image_path = find_image_path(box_path)
        if image_path is None:
            continue
        text = gt_path.read_text(encoding="utf-8").rstrip("\n")
        entries = [entry for entry in load_box_entries(box_path) if entry.char != " "]
        if not entries:
            continue
        with Image.open(image_path) as img:
            height = img.height
        new_entries = insert_spaces(entries, text, height)
        if new_entries is None:
            continue
        save_box_entries(box_path, new_entries)
        updated += 1
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add missing space entries to Baybayin phrase box files.")
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory containing phrase box/GT pairs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"{root} is not a directory")
    updated = process_directory(root)
    print(f"Added spaces to {updated} box files under {root}")


if __name__ == "__main__":
    main()
