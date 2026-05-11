#!/usr/bin/env python3
"""Tighten bounding boxes for characters without diacritics."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image


def load_label_map(script_path: Path) -> Dict[str, str]:
    spec = importlib.util.spec_from_file_location("rebox_archive_dataset", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.LABEL_MAP  # type: ignore[attr-defined]


def tighten_box(tif_path: Path, label: str, threshold: int) -> bool:
    with Image.open(tif_path) as img:
        arr = np.array(img.convert("L"))
    mask = arr < threshold
    if not mask.any():
        return False
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    top = int(rows[0])
    bottom = int(rows[-1] + 1)
    left = int(cols[0])
    right = int(cols[-1] + 1)
    height = arr.shape[0]
    line = f"{label} {left} {height - bottom} {right} {height - top} 0"
    tif_path.with_suffix(".box").write_text(line + "\n", encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Tighten boxes for characters without diacritics.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("__pycache__/char_no_dupe"),
        help="Directory with <prefix>_####.tif/.box files.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=235,
        help="Ink detection threshold.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    label_map = load_label_map(script_dir / "rebox_archive_dataset.py")

    total = 0
    updated = 0
    for tif_path in sorted(args.root.glob('*.tif')):
        stem = tif_path.stem
        if '_' not in stem:
            continue
        prefix = stem.split('_', 1)[0]
        label = label_map.get(prefix)
        if (not label or len(label) != 1) and tif_path.with_suffix('.box').exists():
            first_line = tif_path.with_suffix('.box').read_text(encoding='utf-8').splitlines()
            label = first_line[0].split()[0] if first_line else None
        if not label or len(label) != 1:
            continue
        total += 1
        if tighten_box(tif_path, label, args.threshold):
            updated += 1

    print(f"Tightened {updated}/{total} single-character samples.")


if __name__ == "__main__":
    main()
