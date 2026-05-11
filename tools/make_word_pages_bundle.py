#!/usr/bin/env python3
"""
Build a multi-page TIFF bundle and aggregated BOX file for handwritten Baybayin words.
Targets datasets produced by preprocess_hw_words.py (PNG + BOX + GT per word).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from PIL import Image


def gather_images(root: Path) -> List[Path]:
    images: List[Path] = []
    for path in root.glob("*.png"):
        stem = path.stem
        if "_PNG" in stem:
            continue
        images.append(path)
    return sorted(images)


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("L")


def build_bundle(images: List[Path], out_tif: Path, out_box: Path) -> Tuple[int, int]:
    if not images:
        return 0, 0

    first_image = load_image(images[0])
    append_images: List[Image.Image] = []

    try:
        for path in images[1:]:
            append_images.append(load_image(path))

        first_image.save(
            out_tif,
            save_all=True,
            append_images=append_images,
            compression="tiff_deflate",
        )
    finally:
        first_image.close()
        for img in append_images:
            img.close()

    box_lines: List[str] = []
    total_boxes = 0

    for page_idx, path in enumerate(images):
        box_path = path.with_suffix(".box")
        if not box_path.exists():
            continue
        for raw_line in box_path.read_text(encoding="utf-8").splitlines():
            parts = raw_line.strip().split()
            if len(parts) < 5:
                continue
            ch = parts[0]
            left, bottom, right, top = parts[1:5]
            box_lines.append(f"{ch} {left} {bottom} {right} {top} {page_idx}")
            total_boxes += 1

    if box_lines:
        out_box.write_text("\n".join(box_lines) + "\n", encoding="utf-8")
    else:
        out_box.write_text("", encoding="utf-8")

    return len(images), total_boxes


def main() -> None:
    parser = argparse.ArgumentParser(description="Create page bundles for Baybayin word datasets.")
    parser.add_argument("--root", type=Path, required=True, help="Directory containing PNG + BOX files.")
    parser.add_argument("--output-prefix", type=Path, required=True, help="Destination prefix for the bundle (without extension).")
    args = parser.parse_args()

    root = args.root.resolve()
    images = gather_images(root)

    out_prefix = args.output_prefix.resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    out_tif = out_prefix.with_suffix(".tif")
    out_box = out_prefix.with_suffix(".box")

    pages, boxes = build_bundle(images, out_tif, out_box)
    print(f"Built bundle: {pages} pages, {boxes} boxes -> {out_tif}")


if __name__ == "__main__":
    main()
