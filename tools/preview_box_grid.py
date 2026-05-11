#!/usr/bin/env python3
"""
Render a grid preview of many pages with bounding boxes.

Example:
    python3 tools/preview_box_grid.py \
        --tif final_version/handwritten/characters/ge_gi.tif \
        --box final_version/handwritten/characters/ge_gi.box \
        --start 0 \
        --count 100 \
        --columns 10 \
        --scale 0.75 \
        --output ge_gi_preview.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a grid visualization for multiple boxed pages.")
    parser.add_argument("--tif", type=Path, required=True, help="Path to multipage TIFF.")
    parser.add_argument("--box", type=Path, required=True, help="Path to BOX file.")
    parser.add_argument("--start", type=int, default=0, help="First page index to include.")
    parser.add_argument("--count", type=int, default=100, help="Number of pages to render.")
    parser.add_argument("--columns", type=int, default=10, help="Number of columns in the grid.")
    parser.add_argument(
        "--scale", type=float, default=0.75, help="Scale factor applied to each page before tiling."
    )
    parser.add_argument("--padding", type=int, default=8, help="Padding (pixels) between tiles.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("box_grid_preview.png"),
        help="Destination PNG for the rendered grid.",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=14,
        help="Font size for the page label overlay (depends on font availability).",
    )
    return parser.parse_args()


def load_boxes(box_path: Path) -> Dict[int, List[Tuple[str, int, int, int, int]]]:
    page_map: Dict[int, List[Tuple[str, int, int, int, int]]] = {}
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
        page_map.setdefault(page, []).append((label, left, bottom, right, top))
    return page_map


def get_font(font_size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", font_size)
    except (OSError, IOError):
        return ImageFont.load_default()


def render_page_with_boxes(
    image: Image.Image,
    boxes: List[Tuple[str, int, int, int, int]],
    font: ImageFont.ImageFont,
) -> Image.Image:
    rgb = image.convert("RGB")
    draw = ImageDraw.Draw(rgb)
    width, height = rgb.size

    for label, left, bottom, right, top in boxes:
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

    boxes_by_page = load_boxes(args.box)
    font = get_font(args.font_size)
    padding = args.padding

    pages_to_render = list(range(args.start, args.start + max(0, args.count)))

    rendered_tiles: List[Tuple[int, Image.Image]] = []
    with Image.open(args.tif) as img:
        for page_idx in pages_to_render:
            try:
                img.seek(page_idx)
            except EOFError:
                break
            base = img.copy()
            boxes = boxes_by_page.get(page_idx, [])
            rendered = render_page_with_boxes(base, boxes, font)
            if args.scale != 1.0:
                new_size = (
                    max(1, int(rendered.width * args.scale)),
                    max(1, int(rendered.height * args.scale)),
                )
                rendered = rendered.resize(new_size, Image.NEAREST)
            draw = ImageDraw.Draw(rendered)
            label_text = f"Page {page_idx}"
            draw.rectangle(
                [(0, 0), (draw.textlength(label_text, font=font) + 8, font.size + 8)],
                fill=(0, 0, 0, 128),
            )
            draw.text((4, 4), label_text, fill="white", font=font)
            rendered_tiles.append((page_idx, rendered))

    if not rendered_tiles:
        raise RuntimeError("No pages rendered; check start/count parameters.")

    cols = max(1, args.columns)
    rows = math.ceil(len(rendered_tiles) / cols)
    tile_w = rendered_tiles[0][1].width
    tile_h = rendered_tiles[0][1].height

    grid_w = cols * tile_w + (cols + 1) * padding
    grid_h = rows * tile_h + (rows + 1) * padding

    grid = Image.new("RGB", (grid_w, grid_h), color=(30, 30, 30))

    for idx, (_, tile) in enumerate(rendered_tiles):
        row = idx // cols
        col = idx % cols
        x = padding + col * (tile_w + padding)
        y = padding + row * (tile_h + padding)
        grid.paste(tile, (x, y))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.output)
    print(f"Wrote grid preview with {len(rendered_tiles)} page(s) to {args.output}")


if __name__ == "__main__":
    main()
