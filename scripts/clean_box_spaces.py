#!/usr/bin/env python3
"""
Remove space annotations from Tesseract .box files.

Every .box file is scanned recursively under the provided folder. Lines whose
character label is a literal space are dropped while the rest of the content is
left untouched. A short summary with per-file removal counts is printed once the
run finishes.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple


@dataclass
class FileReport:
    """Holds processing information for a single .box file."""

    path: Path
    removed_spaces: int
    warnings: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove space annotations from every .box file under a folder."
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Folder containing .box files (processed recursively)",
    )
    return parser.parse_args()


def iter_box_files(root: Path) -> Sequence[Path]:
    """Yield every .box file underneath root."""
    return sorted(p for p in root.rglob("*.box") if p.is_file())


def _relative_display(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _extract_char_and_tokens(line: str) -> Tuple[str | None, List[str]]:
    """Return the character label and the remaining numeric tokens."""
    if not line:
        return None, []
    first_char = line[0]
    if first_char.isspace():
        char_token = first_char
        remainder = line[1:]
    else:
        idx = 0
        while idx < len(line) and not line[idx].isspace():
            idx += 1
        char_token = line[:idx]
        remainder = line[idx:]
    tokens = remainder.split()
    return char_token, tokens


def clean_box_file(path: Path) -> Tuple[int, List[str], bool]:
    """Remove space annotation lines from a single .box file."""
    removed = 0
    warnings: List[str] = []
    kept_lines: List[str] = []
    had_bom = False

    try:
        with path.open("r", encoding="utf-8") as handle:
            for lineno, raw_line in enumerate(handle, 1):
                line_body = raw_line.rstrip("\r\n")
                newline = raw_line[len(line_body) :]

                parse_target = line_body
                if lineno == 1 and parse_target.startswith("\ufeff"):
                    had_bom = True
                    parse_target = parse_target[1:]

                if not parse_target:
                    kept_lines.append(parse_target + newline)
                    continue

                char_token, tokens = _extract_char_and_tokens(parse_target)
                if not char_token:
                    warnings.append(f"line {lineno}: unable to read character column")
                    kept_lines.append(parse_target + newline)
                    continue

                if len(tokens) < 5:
                    warnings.append(
                        f"line {lineno}: expected 5 numeric columns, found {len(tokens)}"
                    )
                    kept_lines.append(parse_target + newline)
                    continue

                try:
                    [int(tok) for tok in tokens[:5]]
                except ValueError:
                    warnings.append(
                        f"line {lineno}: coordinate columns must be integers "
                        f"(found: {' '.join(tokens[:5])})"
                    )
                    kept_lines.append(parse_target + newline)
                    continue

                if char_token == " ":
                    removed += 1
                    continue

                kept_lines.append(parse_target + newline)
    except UnicodeDecodeError as exc:
        warnings.append(f"unable to read file as UTF-8: {exc}")
        return removed, warnings, False

    if removed == 0:
        return removed, warnings, False

    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8", newline="") as tmp_handle:
            if had_bom:
                tmp_handle.write("\ufeff")
            tmp_handle.writelines(kept_lines)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return removed, warnings, True


def process(root: Path) -> List[FileReport]:
    reports: List[FileReport] = []
    box_files = iter_box_files(root)

    if not box_files:
        print(f"No .box files found under {root}", file=sys.stderr)
        return reports

    for box in box_files:
        removed, warnings, _ = clean_box_file(box)
        reports.append(FileReport(box, removed, warnings))
    return reports


def print_summary(root: Path, reports: Sequence[FileReport]) -> None:
    total_removed = sum(r.removed_spaces for r in reports)
    print(f"Processed {len(reports)} .box files under {root}")
    for report in reports:
        rel_path = _relative_display(report.path, root)
        print(f"- {rel_path}: removed {report.removed_spaces} space annotations")
        for warn in report.warnings:
            print(f"    warning: {warn}")
    print(f"Total removed space annotations: {total_removed}")


def main() -> int:
    args = parse_args()
    root = args.root.expanduser()

    if not root.exists() or not root.is_dir():
        print(f"Path {root} does not exist or is not a directory.", file=sys.stderr)
        return 1

    reports = process(root)
    if not reports:
        return 0
    print_summary(root, reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
