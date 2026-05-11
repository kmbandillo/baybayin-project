#!/usr/bin/env python3
"""Adjust space box X coordinates to align with the previous main glyph."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


MARKS = {"ᜒ", "ᜓ", "᜔"}


@dataclass
class BoxLine:
    char: str
    left: int
    bottom: int
    right: int
    top: int
    page: int
    raw: str

    @property
    def width(self) -> int:
        return self.right - self.left

    def formatted(self) -> str:
        return f"{self.char} {self.left} {self.bottom} {self.right} {self.top} {self.page}"


def parse_box_line(line: str) -> Optional[BoxLine]:
    stripped = line.rstrip("\n")
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
    return BoxLine(char, left, bottom, right, top, page, stripped)


def find_previous_main(entries: List[BoxLine], start: int) -> Optional[BoxLine]:
    for idx in range(start, -1, -1):
        candidate = entries[idx]
        if candidate.char == " " or candidate.char == "\\n":
            continue
        if candidate.char in MARKS:
            continue
        if not candidate.char.strip():
            continue
        return candidate
    return None


def find_next_main(entries: List[BoxLine], start: int) -> Optional[BoxLine]:
    for idx in range(start, len(entries)):
        candidate = entries[idx]
        if candidate.char == " " or candidate.char == "\\n":
            continue
        if candidate.char in MARKS:
            continue
        if not candidate.char.strip():
            continue
        return candidate
    return None


def adjust_space_boxes(entries: List[BoxLine]) -> bool:
    changed = False
    for idx, entry in enumerate(entries):
        if entry.char != " ":
            continue
        prev = find_previous_main(entries, idx - 1)
        if prev is None:
            continue
        next_main = find_next_main(entries, idx + 1)
        desired_left = prev.right
        desired_right = next_main.left if next_main is not None else entry.right
        if desired_right <= desired_left:
            desired_right = desired_left + max(1, entry.width)
        if entry.left == desired_left and entry.right == desired_right:
            continue
        entry.left = desired_left
        entry.right = desired_right
        changed = True
    return changed


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


def fix_directory(root: Path) -> int:
    adjusted = 0
    for box_path in sorted(root.rglob("*.box")):
        entries = load_box_entries(box_path)
        if not entries:
            continue
        if adjust_space_boxes(entries):
            save_box_entries(box_path, entries)
            adjusted += 1
    return adjusted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align space box X positions to previous main glyphs.")
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory containing .box files (processed recursively).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"{root} is not a directory")
    adjusted = fix_directory(root)
    print(f"Updated {adjusted} box files under {root}")


if __name__ == "__main__":
    main()
