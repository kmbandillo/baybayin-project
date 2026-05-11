#!/usr/bin/env python3
"""
Quick visualization helper for Baybayin box files.

Example:
    python3 tools/visualize_boxes.py \
        --tif final_version/handwritten/characters/ge_gi.tif \
        --box final_version/handwritten/characters/ge_gi.box \
        --pages 0 1 2 \
        --output-dir tmp/visuals
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render bounding boxes over a TIFF page.")
    parser.add_argument("--tif", type=Path, required=True, help="Path to multipage TIFF.")
    parser.add_argument("--box", type=Path, required=True, help="Path to BOX file.")
    parser.add_argument(
        "--pages",
        type=int,
        nargs="*",
        default=[0],
        help="Optional list of page indexes to render (defaults to the first page).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("box_visualizations"),
        help="Directory where rendered PNGs will be written.",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=14,
        help="Font size for drawing labels (depends on font availability).",
    )
    return parser.parse_args()


def load_boxes(box_path: Path) -> List[Tuple[str, int, int, int, int, int]]:
    data: List[Tuple[str, int, int, int, int, int]] = []
    for raw_line in box_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 6:
            continue
        label = parts[0]
        left, bottom, right, top = map(int, parts[1:5])
        try:
            page = int(parts[5])
        except ValueError:
            continue
        data.append((label, left, bottom, right, top, page))
    return data


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_font(font_size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", font_size)
    except (OSError, IOError):
        return ImageFont.load_default()


def render_page(
    image: Image.Image,
    boxes: Iterable[Tuple[str, int, int, int, int]],
    font: ImageFont.ImageFont,
) -> Image.Image:
    rgb = image.convert("RGB")
    draw = ImageDraw.Draw(rgb)
    width, height = rgb.size

    for label, left, bottom, right, top in boxes:
        # Convert from Tesseract bottom-origin to Pillow top-origin coordinates.
        x0 = left
        y0 = height - top
        x1 = right
        y1 = height - bottom
        draw.rectangle([(x0, y0), (x1, y1)], outline="red", width=2)
        draw.text((x0 + 2, y0 + 2), label, fill="yellow", font=font)
    return rgb


def main() -> None:
    args = parse_args()

    if not args.tif.exists():
        raise FileNotFoundError(f"Missing TIFF: {args.tif}")
    if not args.box.exists():
        raise FileNotFoundError(f"Missing BOX: {args.box}")

    all_boxes = load_boxes(args.box)
    page_map: dict[int, List[Tuple[str, int, int, int, int]]] = {}
    for label, left, bottom, right, top, page in all_boxes:
        page_map.setdefault(page, []).append((label, left, bottom, right, top))

    ensure_output_dir(args.output_dir)
    font = get_font(args.font_size)

    with Image.open(args.tif) as img:
        for page in args.pages:
            try:
                img.seek(page)
            except EOFError:
                print(f"Page {page} out of range, skipping.")
                continue

            boxes = page_map.get(page, [])
            rendered = render_page(img, boxes, font)
            out_path = args.output_dir / f"{args.tif.stem}_page_{page:04d}.png"
            rendered.save(out_path)
            print(f"Wrote {out_path} with {len(boxes)} box(es).")


if __name__ == "__main__":
    main()
