#!/usr/bin/env python3
"""
Split a multi-page TIFF + BOX bundle back into individual TIFF/BOX pairs while preserving
box coordinates. File names are taken from the target directory (e.g. existing .gt.txt files)
so the outputs align with the expected word ordering.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from PIL import Image


def gather_names(source_dir: Path) -> List[str]:
    """Collect base names (without the .gt.txt suffix) for deterministic ordering."""
    stems = set()
    for path in source_dir.glob("*.gt.txt"):
        name = path.name
        if not name.endswith(".gt.txt"):
            continue
        stems.add(name[: -len(".gt.txt")])
    if not stems:
        raise ValueError(f"No .gt.txt files found in {source_dir}")
    return sorted(stems)


def load_box_map(box_path: Path) -> Dict[int, List[str]]:
    """Group box lines by page index, dropping the trailing page column."""
    mapping: Dict[int, List[str]] = {}
    if not box_path.exists():
        raise FileNotFoundError(box_path)

    for raw_line in box_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        try:
            page = int(parts[-1])
        except ValueError as exc:
            raise ValueError(f"Invalid page index in line: {raw_line!r}") from exc
        mapping.setdefault(page, []).append(" ".join(parts[:-1]))
    return mapping


def split_bundle(bundle_tif: Path, bundle_box: Path, names_dir: Path, out_dir: Path) -> None:
    names = gather_names(names_dir)
    box_map = load_box_map(bundle_box)

    out_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(bundle_tif) as bundle:
        total_pages = getattr(bundle, "n_frames", 1)
        if total_pages != len(names):
            raise ValueError(f"Page count ({total_pages}) does not match number of names ({len(names)}).")

        for idx, name in enumerate(names):
            bundle.seek(idx)
            frame = bundle.copy()
            single_image = frame.convert("L")
            out_tif = out_dir / f"{name}.tif"
            single_image.save(out_tif, compression="tiff_deflate")
            single_image.close()
            frame.close()

            lines = box_map.get(idx, [])
            out_box = out_dir / f"{name}.box"
            out_box.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split Baybayin word bundle into per-sample TIFF/BOX files.")
    parser.add_argument("--bundle-tif", type=Path, required=True, help="Input multi-page TIFF bundle.")
    parser.add_argument("--bundle-box", type=Path, required=True, help="Input BOX file with page indices.")
    parser.add_argument("--names-dir", type=Path, required=True, help="Directory containing .gt.txt files for naming.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination directory for per-page TIFF/BOX files.")
    args = parser.parse_args()

    split_bundle(
        bundle_tif=args.bundle_tif.resolve(),
        bundle_box=args.bundle_box.resolve(),
        names_dir=args.names_dir.resolve(),
        out_dir=args.output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
