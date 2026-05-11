#!/usr/bin/env python3
"""Detect samples where base glyph and diacritic form a single connected component."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image


def load_label_map(script_path: Path) -> Dict[str, str]:
    spec = importlib.util.spec_from_file_location("rebox_archive_dataset", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.LABEL_MAP  # type: ignore[attr-defined]


def parse_box(path: Path) -> List[Tuple[str, int, int, int, int]]:
    entries: List[Tuple[str, int, int, int, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            left, bottom, right, top = map(int, parts[1:5])
        except ValueError:
            continue
        entries.append((parts[0], left, bottom, right, top))
    return entries


def box_to_slices(left: int, bottom: int, right: int, top: int, height: int, width: int) -> Tuple[slice, slice]:
    row_top = height - top
    row_bottom = height - bottom
    row_top = max(0, min(height, row_top))
    row_bottom = max(0, min(height, row_bottom))
    if row_top > row_bottom:
        row_top, row_bottom = row_bottom, row_top
    col_left = max(0, min(left, right))
    col_right = max(0, min(width, max(left, right)))
    return slice(row_top, row_bottom), slice(col_left, col_right)


def count_components(mask: np.ndarray) -> int:
    visited = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape
    components = 0
    stack: List[Tuple[int, int]] = []
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for row in range(height):
        for col in range(width):
            if not mask[row, col] or visited[row, col]:
                continue
            components += 1
            stack.append((row, col))
            visited[row, col] = True
            while stack:
                cy, cx = stack.pop()
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
    return components


def analyze_sample(
    tif_path: Path, box_path: Path, base_char: str, diac_char: str, threshold: int
) -> bool:
    entries = parse_box(box_path)
    base_bbox = None
    diac_bbox = None
    for ch, left, bottom, right, top in entries:
        if ch == base_char and base_bbox is None:
            base_bbox = (left, bottom, right, top)
        elif ch == diac_char and diac_bbox is None:
            diac_bbox = (left, bottom, right, top)
        if base_bbox and diac_bbox:
            break
    if base_bbox is None or diac_bbox is None:
        return False

    with Image.open(tif_path) as img:
        arr = np.array(img.convert("L"))
    ink = arr < threshold
    combined = np.zeros_like(ink, dtype=bool)

    height, width = ink.shape
    for left, bottom, right, top in (base_bbox, diac_bbox):
        row_slice, col_slice = box_to_slices(left, bottom, right, top, height, width)
        if row_slice.start == row_slice.stop or col_slice.start == col_slice.stop:
            continue
        combined[row_slice, col_slice] |= ink[row_slice, col_slice]

    if not combined.any():
        return False
    components = count_components(combined)
    return components <= 1


def derive_prefix_from_name(path: Path) -> str | None:
    stem = path.stem
    if "_" not in stem:
        return None
    prefix, suffix = stem.rsplit("_", 1)
    if suffix.isdigit():
        return prefix
    return None


def iter_samples(root: Path, prefixes: Iterable[str]) -> Iterable[Tuple[str, Path]]:
    prefixes_set = set(prefixes)
    has_files = any(child.is_file() and child.suffix.lower() == ".tif" for child in root.iterdir())

    if has_files:
        for tif_path in sorted(root.glob("*.tif")):
            prefix = derive_prefix_from_name(tif_path)
            if prefix is None or prefix not in prefixes_set:
                continue
            yield prefix, tif_path
        return

    for prefix_dir in sorted(root.iterdir()):
        if not prefix_dir.is_dir() or prefix_dir.name not in prefixes_set:
            continue
        for tif_path in sorted(prefix_dir.glob("*.tif")):
            yield prefix_dir.name, tif_path


def delete_sample(tif_path: Path, extra_suffixes: Sequence[str]) -> None:
    base = tif_path.with_suffix("")
    for suffix in (".tif", ".box", *extra_suffixes):
        target = Path(str(base) + suffix)
        if target.exists():
            target.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect attached diacritics in per-sample character sets.")
    parser.add_argument("--root", type=Path, required=True, help="Directory containing prefix subfolders with TIFF/BOX pairs.")
    parser.add_argument("--threshold", type=int, default=235, help="Binary threshold for ink detection")
    parser.add_argument(
        "--max-print",
        type=int,
        default=5,
        help="Maximum number of sample paths to list per prefix (default: 5, 0 to disable).",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete flagged samples (.tif/.box + common auxiliary suffixes) instead of just reporting them.",
    )
    parser.add_argument(
        "--extra-suffix",
        nargs="*",
        default=[".gt.txt", ".lstmf"],
        help="Extra suffixes to delete when --delete is used.",
    )
    args = parser.parse_args()

    label_map = load_label_map(Path(__file__).with_name("rebox_archive_dataset.py"))

    prefixes = [p for p, label in label_map.items() if len(label) >= 2]

    problematic: List[Tuple[str, Path]] = []
    for prefix, tif_path in iter_samples(args.root, prefixes):
        label = label_map[prefix]
        base_char, diac_char = label[0], label[1]
        box_path = tif_path.with_suffix(".box")
        if not box_path.exists():
            continue
        try:
            attached = analyze_sample(tif_path, box_path, base_char, diac_char, args.threshold)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Failed on {tif_path}: {exc}")
            continue
        if attached:
            problematic.append((prefix, tif_path))

    print(f"Found {len(problematic)} samples with attached diacritics in {args.root}.")
    if not problematic:
        return

    per_prefix: Dict[str, List[Path]] = {}
    for prefix, path in problematic:
        per_prefix.setdefault(prefix, []).append(path)

    for prefix in sorted(per_prefix):
        samples = per_prefix[prefix]
        print(f"  {prefix}: {len(samples)} samples")
        if args.max_print > 0:
            for path in samples[: args.max_print]:
                print(f"    - {path}")

    if args.delete:
        for _, path in problematic:
            delete_sample(path, args.extra_suffix)
        print(f"Deleted {len(problematic)} samples from {args.root}.")


if __name__ == "__main__":
    main()
