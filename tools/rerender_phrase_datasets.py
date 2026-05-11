#!/usr/bin/env python3
"""Re-render Baybayin phrase datasets with precise boxes using HarfBuzz."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from generate_boxes_and_lstmf import convert_boxes_to_entries, write_box_file
from render_baybayin_namin import (
    BaybayinTextRenderer,
    segment_baybayin,
    transliterate_to_font_encoding,
)


def normalise_phrase(text: str) -> str:
    # Collapse all whitespace runs to single spaces, strip trailing newlines.
    tokens = text.strip().split()
    return " ".join(tokens)


def save_image_pair(image: Image.Image, base_path: Path) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    png_path = base_path.with_suffix(".png")
    tif_path = base_path.with_suffix(".tif")
    image.save(png_path)
    image.save(tif_path, format="TIFF", compression="tiff_deflate")


def process_dir(
    name: str,
    directory: Path,
    renderer: BaybayinTextRenderer,
    *,
    font_encoding: str,
    group_marks: bool,
) -> Tuple[int, int]:
    updated = 0
    skipped = 0

    for gt_path in sorted(directory.glob("*.gt.txt")):
        raw_text = gt_path.read_text(encoding="utf-8")
        text = normalise_phrase(raw_text)
        if not text:
            skipped += 1
            continue

        if font_encoding == "latin":
            font_text = transliterate_to_font_encoding(text)
        else:
            font_text = text

        image, box_records = renderer.render_with_boxes(font_text)
        boxes = [(x0, y0, x1, y1) for (x0, y0, x1, y1, _cluster) in box_records]
        tokens = segment_baybayin(text, group_marks=group_marks)
        try:
            entries = convert_boxes_to_entries(boxes, tokens, image.size)
        except ValueError as exc:
            print(f"[{name}] WARN {gt_path.name}: {exc}")
            skipped += 1
            continue

        base = gt_path.with_suffix("")
        save_image_pair(image, base)
        write_box_file(base.with_suffix(".box"), entries)
        base.with_suffix(".gt.txt").write_text(text + "\n", encoding="utf-8")
        txt_path = base.with_suffix(".txt")
        if txt_path.exists():
            txt_path.write_text(text + "\n", encoding="utf-8")
        else:
            txt_path.write_text(text + "\n", encoding="utf-8")

        updated += 1
        if updated % 50 == 0:
            print(f"[{name}] updated {updated} samples…")

    return updated, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerender Baybayin phrase datasets with new boxes.")
    parser.add_argument(
        "--bn-dir",
        type=Path,
        default=Path("baybayin_dataset/phrase/bn_phrase"),
        help="Directory containing Baybayin Namin phrase files.",
    )
    parser.add_argument(
        "--ts-dir",
        type=Path,
        default=Path("baybayin_dataset/phrase/ts_phrase"),
        help="Directory containing Tagalog Stylized phrase files.",
    )
    parser.add_argument(
        "--bn-font",
        type=Path,
        default=Path("baybayin_dataset/BaybayinNamin.otf"),
        help="Font path for Baybayin Namin phrases.",
    )
    parser.add_argument(
        "--ts-font",
        type=Path,
        default=Path("baybayin_dataset/TagalogStylized.ttf"),
        help="Font path for Tagalog Stylized phrases.",
    )
    parser.add_argument("--font-size", type=int, default=128, help="Font size for rendering.")
    parser.add_argument("--margin", type=int, default=48, help="Margin (pixels) around text.")
    parser.add_argument("--line-spacing", type=float, default=1.35, help="Line spacing multiplier.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.bn_dir.exists():
        bn_renderer = BaybayinTextRenderer(
            font_path=args.bn_font,
            font_size=args.font_size,
            margin=args.margin,
            line_spacing=args.line_spacing,
        )
        updated, skipped = process_dir(
            "bn",
            args.bn_dir,
            bn_renderer,
            font_encoding="latin",
            group_marks=True,
        )
        print(f"[bn] completed {updated} files (skipped {skipped}).")
    else:
        print(f"[bn] directory {args.bn_dir} missing; skipping.")

    if args.ts_dir.exists():
        ts_renderer = BaybayinTextRenderer(
            font_path=args.ts_font,
            font_size=args.font_size,
            margin=args.margin,
            line_spacing=args.line_spacing,
        )
        updated, skipped = process_dir(
            "ts",
            args.ts_dir,
            ts_renderer,
            font_encoding="baybayin",
            group_marks=False,
        )
        print(f"[ts] completed {updated} files (skipped {skipped}).")
    else:
        print(f"[ts] directory {args.ts_dir} missing; skipping.")


if __name__ == "__main__":
    main()
