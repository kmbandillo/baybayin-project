#!/usr/bin/env python3
"""
Render Tagalog Stylized phrase images using the Baybayin Namin phrase GT text.
Produces PNG placeholders (which can be reboxed later) plus companion .txt files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Tagalog Stylized phrases from existing GT text.")
    parser.add_argument(
        "--bn-dir",
        type=Path,
        required=True,
        help="Directory containing baybayin_namin_phrase_*.gt.txt files.",
    )
    parser.add_argument(
        "--ts-dir",
        type=Path,
        required=True,
        help="Output directory for Tagalog stylized phrases.",
    )
    parser.add_argument(
        "--font",
        type=Path,
        default=Path("final_version/TagalogStylized.ttf"),
        help="Path to Tagalog stylized TTF font.",
    )
    parser.add_argument("--font-size", type=int, default=96, help="Font size for rendering.")
    parser.add_argument("--padding", type=int, default=32, help="Padding around the text in pixels.")
    return parser.parse_args()


def render_text_image(text: str, font: ImageFont.FreeTypeFont, padding: int) -> Image.Image:
    bbox = font.getbbox(text)
    width = max(1, bbox[2] - bbox[0] + padding * 2)
    height = max(1, bbox[3] - bbox[1] + padding * 2)
    image = Image.new("L", (width, height), color=255)
    draw = ImageDraw.Draw(image)
    draw.text((padding - bbox[0], padding - bbox[1]), text, font=font, fill=0)
    return image


def main() -> None:
    args = parse_args()
    bn_dir = args.bn_dir
    ts_dir = args.ts_dir
    ts_dir.mkdir(parents=True, exist_ok=True)

    font = ImageFont.truetype(str(args.font), args.font_size)

    generated = 0
    for gt_path in sorted(bn_dir.glob("*.gt.txt")):
        text = gt_path.read_text(encoding="utf-8").rstrip("\n")
        if not text:
            continue
        stem = gt_path.stem.replace(".gt", "") if gt_path.stem.endswith(".gt") else gt_path.stem
        out_png = ts_dir / f"{stem}.png"
        out_txt = ts_dir / f"{stem}.txt"
        out_font = ts_dir / f"{stem}.font.txt"

        image = render_text_image(text, font, args.padding)
        image.save(out_png)
        out_txt.write_text(text + "\n", encoding="utf-8")
        out_font.write_text("TagalogStylized\n", encoding="utf-8")
        generated += 1

    print(f"Rendered {generated} phrases into {ts_dir}")


if __name__ == "__main__":
    main()
