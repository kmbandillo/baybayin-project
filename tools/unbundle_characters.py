#!/usr/bin/env python3
"""
Unbundle multi-page character TIFF/BOX files into individual samples.

Example:
    python3 tools/unbundle_characters.py \
        --bundle-dir final_version/archive_bundle \
        --output-dir __pycache__/char_unbundled/characters
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageSequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split bundled character TIFF/BOX files.")
    parser.add_argument("--bundle-dir", type=Path, required=True, help="Directory containing bundled .tif/.box files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write unbundled files.")
    return parser.parse_args()


def load_boxes(box_path: Path) -> Dict[int, List[str]]:
    per_page: Dict[int, List[str]] = {}
    for line in box_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        page = int(parts[5])
        per_page.setdefault(page, []).append(line)
    return per_page


def write_sample(output_dir: Path, base: str, page_idx: int, image: Image.Image, lines: List[str]) -> None:
    stem = f"{base}_{page_idx:04d}"
    out_tif = output_dir / f"{stem}.tif"
    out_box = output_dir / f"{stem}.box"
    out_gt = output_dir / f"{stem}.gt.txt"

    image.save(out_tif)

    cleaned_lines: List[str] = []
    tokens: List[str] = []
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        tokens.append(parts[0])
        cleaned_lines.append(" ".join(parts[:5] + ["0"]))

    out_box.write_text("\n".join(cleaned_lines) + "\n", encoding="utf-8")
    out_gt.write_text("".join(tokens) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    bundle_dir = args.bundle_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for tif_path in sorted(bundle_dir.glob("*.tif")):
        box_path = tif_path.with_suffix(".box")
        if not box_path.exists():
            continue
        per_page = load_boxes(box_path)
        with Image.open(tif_path) as img:
            for page_idx, frame in enumerate(ImageSequence.Iterator(img)):
                if page_idx not in per_page:
                    continue
                out_img = frame.convert("L")
                write_sample(output_dir, tif_path.stem, page_idx, out_img, per_page[page_idx])
                processed += 1
    print(f"Unbundled {processed} samples into {output_dir}")


if __name__ == "__main__":
    main()
