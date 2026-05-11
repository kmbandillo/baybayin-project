#!/usr/bin/env python3
"""
Create multi-page TIFF + BOX bundles for handwritten Baybayin characters grouped by
prefix (before the final underscore) so they can be inspected in JTessBox Editor.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


def gather_images(root: Path) -> Dict[str, List[Tuple[int, Path]]]:
    groups: Dict[str, List[Tuple[int, Path]]] = defaultdict(list)
    for tif_path in sorted(root.glob("*.tif")):
        stem = tif_path.stem
        if "_" not in stem:
            continue
        prefix, idx_str = stem.rsplit("_", 1)
        if not idx_str.isdigit():
            continue
        box_path = tif_path.with_suffix(".box")
        if not box_path.exists() or box_path.stat().st_size == 0:
            continue
        groups[prefix].append((int(idx_str), tif_path))
    return groups


def build_bundle(prefix: str, items: List[Tuple[int, Path]], out_dir: Path) -> Tuple[int, int]:
    items.sort()
    if not items:
        return 0, 0

    images: List[Image.Image] = []
    first_image: Image.Image | None = None

    for _, path in items:
        with Image.open(path) as im:
            im_l = im.convert("L")
            if first_image is None:
                first_image = im_l.copy()
            else:
                images.append(im_l.copy())

    assert first_image is not None

    out_tif = out_dir / f"{prefix}.tif"
    first_image.save(out_tif, save_all=True, append_images=images, compression="tiff_deflate")
    first_image.close()
    for im in images:
        im.close()

    box_lines: List[str] = []
    for page_idx, (_, path) in enumerate(items):
        box_path = path.with_suffix(".box")
        if not box_path.exists() or box_path.stat().st_size == 0:
            continue
        for line in box_path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            ch = parts[0]
            left, bottom, right, top = parts[1:5]
            box_lines.append(f"{ch} {left} {bottom} {right} {top} {page_idx}")

    out_box = out_dir / f"{prefix}.box"
    out_box.write_text("\n".join(box_lines) + "\n", encoding="utf-8")

    return len(items), len(box_lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build JTessBox bundles for Baybayin characters.")
    parser.add_argument("--root", type=Path, default=Path("full_dataset/character"), help="Character image directory")
    parser.add_argument("--output", type=Path, default=Path("full_dataset/pages_bundle"), help="Destination directory")
    args = parser.parse_args()

    root = args.root.resolve()
    out_dir = args.output.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = gather_images(root)
    print(f"Found {len(groups)} character prefixes under {root}.")

    total_pages = 0
    total_boxes = 0
    for prefix, items in sorted(groups.items()):
        pages, boxes = build_bundle(prefix, items, out_dir)
        total_pages += pages
        total_boxes += boxes
        print(f"  {prefix}: {pages} pages, {boxes} boxes -> {out_dir / (prefix + '.tif')}")

    print(f"Completed bundles: {total_pages} pages, {total_boxes} boxes written to {out_dir}.")


if __name__ == "__main__":
    main()
