#!/usr/bin/env python3
"""Rebox per-sample TIFF/BOX pairs, separating main glyph and diacritic."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_label_map(script_dir: Path) -> Dict[str, str]:
    module = load_module(script_dir / "rebox_archive_dataset.py", "rebox_archive_dataset")
    return module.LABEL_MAP  # type: ignore[attr-defined]


def bbox_to_line(ch: str, bbox: tuple[int, int, int, int], height: int) -> str:
    left, top, right, bottom = bbox
    left = max(0, left)
    right = max(left + 1, right)
    top = max(0, top)
    bottom = max(top + 1, bottom)
    return f"{ch} {left} {height - bottom} {right} {height - top} 0"


def rebox_file(
    tif_path: Path,
    orientation: str,
    base_char: str,
    diac_char: str,
    threshold: int,
    helpers,
) -> bool:
    with Image.open(tif_path) as img:
        arr = np.array(img.convert("L"))
    mask = arr < threshold
    base_mask, diac_mask, _ = helpers.split_masks(mask, orientation)
    base_bbox = helpers.bbox_from_mask(base_mask)
    diac_bbox = helpers.bbox_from_mask(diac_mask)
    if base_bbox is None or diac_bbox is None:
        return False
    height = arr.shape[0]
    lines = [
        bbox_to_line(base_char, base_bbox, height),
        bbox_to_line(diac_char, diac_bbox, height),
    ]
    box_path = tif_path.with_suffix(".box")
    box_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebox char_unbundled samples with separate diacritic boxes.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("__pycache__/char_no_dupe"),
        help="Directory containing <prefix>_####.tif/.box files.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=235,
        help="Ink threshold (0-255).",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    label_map = load_label_map(script_dir)
    helpers = load_module(script_dir / "split_b_base_diacritic_boxes.py", "split_helpers")
    diac_positions = helpers.DIAC_POSITIONS

    total = 0
    updated = 0
    skipped = 0
    for prefix_dir in sorted(args.root.iterdir()):
        if not prefix_dir.is_dir():
            continue
        prefix = prefix_dir.name
        label = label_map.get(prefix)
        if not label or len(label) < 2:
            continue
        diac_char = label[1]
        orientation = diac_positions.get(diac_char)
        if orientation is None:
            continue
        base_char = label[0]
<<<<<<< ours
        total += 1
        ok = rebox_file(tif_path, orientation, base_char, diac_char, args.threshold, helpers)
        if ok:
            updated += 1
        else:
            skipped += 1
=======
        for tif_path in sorted(prefix_dir.glob("*.tif")):
            total += 1
            ok = rebox_file(tif_path, orientation, base_char, diac_char, args.threshold, split_helpers)
            if ok:
                updated += 1
            else:
                skipped += 1
>>>>>>> theirs

    print(
        f"Processed {total} TIFFs under {args.root}. Updated {updated}, skipped {skipped} (missing masks)."
    )


if __name__ == "__main__":
    main()
