#!/usr/bin/env python3
"""Split character bundles (multi-page TIF + BOX) into per-sample files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


def parse_box(box_path: Path) -> Dict[int, List[Tuple[str, int, int, int, int]]]:
    page_map: Dict[int, List[Tuple[str, int, int, int, int]]] = {}
    for line in box_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            left, bottom, right, top = map(int, parts[1:5])
        except ValueError:
            continue
        page_idx = int(parts[5]) if len(parts) >= 6 else 0
        page_map.setdefault(page_idx, []).append((parts[0], left, bottom, right, top))
    return page_map


def unbundle(prefix: str, bundle_dir: Path, out_dir: Path) -> Tuple[int, int]:
    tif_path = bundle_dir / f"{prefix}.tif"
    box_path = bundle_dir / f"{prefix}.box"
    if not tif_path.exists() or not box_path.exists():
        return 0, 0

    box_map = parse_box(box_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    page_count = 0
    box_lines = 0
    with Image.open(tif_path) as bundle:
        total_frames = getattr(bundle, "n_frames", 1)
        for page_idx in range(total_frames):
            bundle.seek(page_idx)
            frame = bundle.copy().convert("L")
            stem = f"{prefix}_{page_idx:04d}"
            frame_path = out_dir / f"{stem}.tif"
            frame.save(frame_path, compression="tiff_deflate")
            frame.close()

            lines = box_map.get(page_idx, [])
            box_lines += len(lines)
            box_out = out_dir / f"{stem}.box"
            box_out.write_text(
                "\n".join(
                    f"{ch} {left} {bottom} {right} {top}" for ch, left, bottom, right, top in lines
                )
                + ("\n" if lines else ""),
                encoding="utf-8",
            )
            page_count += 1

    return page_count, box_lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Unbundle character bundles into per-sample TIFF/BOX files.")
    parser.add_argument("--bundle-dir", type=Path, required=True, help="Directory containing <prefix>.tif/.box pairs.")
    parser.add_argument("--output", type=Path, required=True, help="Destination directory for per-sample files.")
    args = parser.parse_args()

    bundle_dir = args.bundle_dir.resolve()
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    total_pages = 0
    total_boxes = 0
    for tif_path in sorted(bundle_dir.glob("*.tif")):
        prefix = tif_path.stem
        pages, boxes = unbundle(prefix, bundle_dir, out_dir)
        total_pages += pages
        total_boxes += boxes
        print(f"{prefix}: wrote {pages} samples, {boxes} box lines -> {out_dir}")

    print(f"Completed unbundling: {total_pages} pages, {total_boxes} box lines written to {out_dir}.")


if __name__ == "__main__":
    main()
