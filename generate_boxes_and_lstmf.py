#!/usr/bin/env python3
"""
Generate .box files and LSTM training data (.lstmf) for the rendered Baybayin Namin dataset.
"""

from __future__ import annotations

import argparse
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from PIL import Image

from render_baybayin_namin import (
    BaybayinTextRenderer,
    segment_baybayin,
    transliterate_to_font_encoding,
)


PSM_MAP = {
    "baybayin_namin_character_images": 10,
    "baybayin_namin_words_images": 7,
    "baybayin_namin_pages_images": 6,
    "baybayin_namin_single": 7,
    "baybayin_namin_multi": 6,
    "tagalog_stylized_single": 7,
    "tagalog_stylized_multi": 6,
}


@dataclass
class BoxEntry:
    char: str
    left: int
    bottom: int
    right: int
    top: int


def convert_boxes_to_entries(
    boxes: List[Tuple[float, float, float, float]],
    tokens: List[str],
    image_size: Tuple[int, int],
) -> List[BoxEntry]:
    entries: List[BoxEntry] = []
    width, height = image_size
    glyph_index = 0

    for token in tokens:
        if token == "\n":
            entries.append(BoxEntry("\\n", 0, 0, 0, 0))
            continue
        if token == " ":
            if glyph_index >= len(boxes):
                raise ValueError("Ran out of glyph boxes while mapping Baybayin tokens.")
            x0, y0, x1, y1 = boxes[glyph_index]
            glyph_index += 1
            left = max(0, int(math.floor(x0)))
            right = min(width, int(math.ceil(x1)))
            top_tl = max(0, int(math.floor(y0)))
            bottom_tl = min(height, int(math.ceil(y1)))
            bottom = max(0, height - bottom_tl)
            top = min(height, height - top_tl)
            entries.append(BoxEntry(" ", left, bottom, right, top))
            continue
        if token.strip() == "":
            continue

        if glyph_index >= len(boxes):
            raise ValueError("Ran out of glyph boxes while mapping Baybayin tokens.")

        x0, y0, x1, y1 = boxes[glyph_index]
        glyph_index += 1

        left = max(0, int(math.floor(x0)))
        right = min(width, int(math.ceil(x1)))
        top_tl = max(0, int(math.floor(y0)))
        bottom_tl = min(height, int(math.ceil(y1)))

        bottom = max(0, height - bottom_tl)
        top = min(height, height - top_tl)

        # Ensure the bounds are sane.
        if right < left:
            right = left
        if top < bottom:
            top = bottom

        for ch in token:
            entries.append(BoxEntry(ch, left, bottom, right, top))

    if glyph_index != len(boxes):
        raise ValueError(
            f"Unmatched glyph boxes: consumed {glyph_index}, but {len(boxes)} available."
        )

    return entries


def write_box_file(path: Path, entries: Iterable[BoxEntry]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(
                f"{entry.char} {entry.left} {entry.bottom} {entry.right} {entry.top} 0\n"
            )


def run_tesseract_lstm(
    image_path: Path,
    output_base: Path,
    lang: str,
    psm: int,
    tessdata_dir: Path,
    config_path: Path,
) -> None:
    lstmf_path = output_base.with_suffix(".lstmf")
    if lstmf_path.exists():
        lstmf_path.unlink()

    cwd = image_path.parent
    cmd = [
        "tesseract",
        image_path.name,
        output_base.name,
        "-l",
        lang,
        "--psm",
        str(psm),
        "--oem",
        "1",
        "--tessdata-dir",
        str(tessdata_dir.resolve()),
        str(config_path.resolve()),
    ]
    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
    )
    if not lstmf_path.exists():
        raise RuntimeError(f"Tesseract did not produce {lstmf_path.name}")


def process_split(
    split_name: str,
    images_dir: Path,
    renderer: BaybayinTextRenderer,
    lang: str,
    tessdata_dir: Path,
    config_path: Path,
    font_encoding: str,
) -> Tuple[int, List[str]]:
    psm = PSM_MAP[split_name]
    failures: List[str] = []
    processed = 0

    tif_paths = sorted(images_dir.glob("*.tif"))
    for tif_path in tif_paths:
        base = tif_path.with_suffix("")
        gt_path = base.with_suffix(".gt.txt")
        if not gt_path.exists():
            failures.append(f"Missing GT for {tif_path.name}")
            continue

        baybayin_text = gt_path.read_text(encoding="utf-8")
        if font_encoding == "latin":
            font_text = transliterate_to_font_encoding(baybayin_text)
        else:
            font_text = baybayin_text
        img_render, boxes = renderer.render_with_boxes(font_text)

        with Image.open(tif_path) as img_file:
            actual_size = img_file.size

        if img_render.size != actual_size:
            failures.append(
                f"Size mismatch for {tif_path.name}: rendered {img_render.size} vs actual {actual_size}"
            )
            continue

        tokens = segment_baybayin(baybayin_text, group_marks=(font_encoding == "latin"))
        try:
            entries = convert_boxes_to_entries(boxes, tokens, img_render.size)
        except ValueError as exc:
            failures.append(f"{tif_path.name}: {exc}")
            continue

        box_path = base.with_suffix(".box")
        write_box_file(box_path, entries)

        try:
            run_tesseract_lstm(
                tif_path,
                base,
                lang=lang,
                psm=psm,
                tessdata_dir=tessdata_dir,
                config_path=config_path,
            )
        except subprocess.CalledProcessError as exc:
            failures.append(
                f"Tesseract failed for {tif_path.name}: {exc.stderr.decode('utf-8', 'ignore')}"
            )
            continue
        except RuntimeError as exc:
            failures.append(f"{tif_path.name}: {exc}")
            continue

        processed += 1
        if processed % 100 == 0:
            print(f"  processed {processed} samples for {split_name}…")

    return processed, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate .box and .lstmf files using pre-rendered Baybayin images."
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=Path("full_dataset/Baybayin_Namin_rendered"),
        help="Root directory that contains *_images folders.",
    )
    parser.add_argument(
        "--font-path",
        type=Path,
        default=Path("full_dataset/Baybayin_Namin/BaybayinNamin.otf"),
        help="Font path used by the renderer (should match the rendered dataset).",
    )
    parser.add_argument("--font-size", type=int, default=128, help="Font size used during rendering.")
    parser.add_argument("--margin", type=int, default=48, help="Margin used during rendering.")
    parser.add_argument("--line-spacing", type=float, default=1.35, help="Line spacing used during rendering.")
    parser.add_argument(
        "--font-encoding",
        choices=("latin", "baybayin"),
        default="latin",
        help="Encoding expected by the font (matches render_baybayin_namin.py).",
    )
    parser.add_argument(
        "--lang",
        default="baybayin_full_best",
        help="Base traineddata to use when generating LSTM features.",
    )
    parser.add_argument(
        "--tessdata-dir",
        type=Path,
        default=Path("releases/baybayin_full_current"),
        help="Directory containing the chosen traineddata file and configs.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("releases/tesseract_training/data/configs/lstm.train"),
        help="Path to the lstm.train config file.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        choices=list(PSM_MAP.keys()),
        default=list(PSM_MAP.keys()),
        help="Which splits to process.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    renderer = BaybayinTextRenderer(
        font_path=args.font_path,
        font_size=args.font_size,
        margin=args.margin,
        line_spacing=args.line_spacing,
    )

    total_processed = 0
    all_failures: List[str] = []

    for split in args.splits:
        images_dir = args.images_root / split
        print(f"[{split}] generating boxes and lstmf for {images_dir}")
        processed, failures = process_split(
            split,
            images_dir,
            renderer,
            lang=args.lang,
            tessdata_dir=args.tessdata_dir,
            config_path=args.config,
            font_encoding=args.font_encoding,
        )
        total_processed += processed
        all_failures.extend(f"{split}/{msg}" for msg in failures)
        print(f"[{split}] processed {processed} images.")

    print(f"Completed with {total_processed} successful samples.")
    if all_failures:
        log_path = Path("lstm_generation_report.txt")
        log_path.write_text("\n".join(all_failures), encoding="utf-8")
        print(f"Encountered {len(all_failures)} issues; details written to {log_path}.")
    else:
        print("No failures encountered.")


if __name__ == "__main__":
    main()
