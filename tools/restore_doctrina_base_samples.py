#!/usr/bin/env python3
"""Restore base-only Doctrina characters from a backup directory."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

BASE_GLYPHS = {
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
}

SUFFIXES = (".gt.txt", ".box", ".png", ".tif")


def iter_base_gt_paths(root: Path) -> Iterable[Path]:
    for gt_path in sorted(root.glob("*.gt.txt")):
        label = gt_path.read_text(encoding="utf-8").strip()
        if label in BASE_GLYPHS:
            yield gt_path


def strip_double_suffix(gt_path: Path) -> Path:
    stem = gt_path.with_suffix("")
    stem = stem.with_suffix("")
    return stem


def copy_sample(stem_src: Path, stem_dst: Path) -> Tuple[int, List[str]]:
    copied = 0
    missing: List[str] = []
    stem_dst.parent.mkdir(parents=True, exist_ok=True)
    for suffix in SUFFIXES:
        src_file = stem_src.with_suffix(suffix)
        if not src_file.exists():
            missing.append(src_file.name)
            continue
        dst_file = stem_dst.with_suffix(suffix)
        dst_file.write_bytes(src_file.read_bytes())
        copied += 1
    return copied, missing


def restore_base_samples(source: Path, dest: Path) -> Tuple[int, int, List[str]]:
    restored = 0
    total = 0
    issues: List[str] = []
    for gt_path in iter_base_gt_paths(source):
        total += 1
        rel_name = gt_path.name
        src_stem = strip_double_suffix(gt_path)
        dst_stem = dest / src_stem.name
        copied, missing = copy_sample(src_stem, dst_stem)
        if copied > 0:
            restored += 1
        if missing:
            issues.append(f"{rel_name}: missing {', '.join(missing)}")
    return restored, total, issues


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy Doctrina base consonant samples from a backup directory."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("dataset/characters/doctrina_characters_noauto_auto_backup"),
        help="Directory containing the backup Doctrina samples.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("dataset/characters/doctrina_characters_noauto"),
        help="Directory where the restored samples should be placed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    restored, total, issues = restore_base_samples(args.source.resolve(), args.dest.resolve())
    print(f"Restored {restored}/{total} base samples from {args.source} into {args.dest}.")
    if issues:
        print("Warnings:")
        for msg in issues:
            print(f"  - {msg}")


if __name__ == "__main__":
    main()
