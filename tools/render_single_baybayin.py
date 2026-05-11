#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from render_baybayin_namin import BaybayinTextRenderer, transliterate_to_font_encoding


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a single Baybayin string to PNG/TIF with GT text.")
    parser.add_argument("text", help="Baybayin Unicode text to render")
    parser.add_argument("out_prefix", type=Path, help="Output path prefix (no extension)")
    parser.add_argument("--font", type=Path, default=Path("full_dataset/Baybayin_Namin/BaybayinNamin.otf"))
    parser.add_argument("--font-size", type=int, default=128)
    parser.add_argument("--margin", type=int, default=48)
    parser.add_argument("--line-spacing", type=float, default=1.35)
    args = parser.parse_args()

    renderer = BaybayinTextRenderer(
        font_path=args.font,
        font_size=args.font_size,
        margin=args.margin,
        line_spacing=args.line_spacing,
    )

    font_text = transliterate_to_font_encoding(args.text)
    image = renderer.render(font_text)

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.out_prefix.with_suffix(".png"))
    image.save(args.out_prefix.with_suffix(".tif"), format="TIFF", compression="tiff_deflate")
    args.out_prefix.with_suffix(".gt.txt").write_text(args.text + "\n", encoding="utf-8")
    args.out_prefix.with_suffix(".latin.txt").write_text(font_text + "\n", encoding="utf-8")
    print(f"Rendered {args.out_prefix}.[png|tif] with text '{args.text}'.")


if __name__ == "__main__":
    main()
