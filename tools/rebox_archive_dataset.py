#!/usr/bin/env python3
"""
Rebox the handwritten archive dataset by generating tight bounding boxes that include
any kudlit or virama marks and writing fresh TIFF/BOX/GT triplets per sample.

The source directory is expected to contain JPEG images named like
    <prefix>.<random>-<writer>.jpg
where <prefix> encodes the Baybayin syllable variant (e.g., ka, ke_ki, k, etc.).

Outputs are written to an explicit destination folder with sanitized filenames
(<prefix>_<index>.tif|box|gt.txt).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

# Unicode marks used in Baybayin
KUDLIT_I = "\u1712"  # ᜒ
KUDLIT_O = "\u1713"  # ᜓ
VIRAMA = "\u1714"  # ᜔


LABEL_MAP: Dict[str, str] = {
    "a": "\u1700",  # ᜀ
    "e_i": "\u1701",  # ᜁ
    "o_u": "\u1702",  # ᜂ
    "b": "\u170a" + VIRAMA,
    "ba": "\u170a",
    "be_bi": "\u170a" + KUDLIT_I,
    "bo_bu": "\u170a" + KUDLIT_O,
    "d": "\u1707" + VIRAMA,
    "da_ra": "\u1707",
    "de_di": "\u1707" + KUDLIT_I,
    "do_du": "\u1707" + KUDLIT_O,
    "g": "\u1704" + VIRAMA,
    "ga": "\u1704",
    "ge_gi": "\u1704" + KUDLIT_I,
    "go_gu": "\u1704" + KUDLIT_O,
    "h": "\u1711" + VIRAMA,
    "ha": "\u1711",
    "he_hi": "\u1711" + KUDLIT_I,
    "ho_hu": "\u1711" + KUDLIT_O,
    "k": "\u1703" + VIRAMA,
    "ka": "\u1703",
    "ke_ki": "\u1703" + KUDLIT_I,
    "ko_ku": "\u1703" + KUDLIT_O,
    "l": "\u170e" + VIRAMA,
    "la": "\u170e",
    "le_li": "\u170e" + KUDLIT_I,
    "lo_lu": "\u170e" + KUDLIT_O,
    "m": "\u170d" + VIRAMA,
    "ma": "\u170d",
    "me_mi": "\u170d" + KUDLIT_I,
    "mo_mu": "\u170d" + KUDLIT_O,
    "n": "\u1708" + VIRAMA,
    "na": "\u1708",
    "ne_ni": "\u1708" + KUDLIT_I,
    "no_nu": "\u1708" + KUDLIT_O,
    "ng": "\u1705" + VIRAMA,
    "nga": "\u1705",
    "nge_ngi": "\u1705" + KUDLIT_I,
    "ngo_ngu": "\u1705" + KUDLIT_O,
    "p": "\u1709" + VIRAMA,
    "pa": "\u1709",
    "pe_pi": "\u1709" + KUDLIT_I,
    "po_pu": "\u1709" + KUDLIT_O,
    "r": "\u1707" + VIRAMA,
    "ra": "\u1707",
    "re_ri": "\u1707" + KUDLIT_I,
    "ro_ru": "\u1707" + KUDLIT_O,
    "s": "\u1710" + VIRAMA,
    "sa": "\u1710",
    "se_si": "\u1710" + KUDLIT_I,
    "so_su": "\u1710" + KUDLIT_O,
    "t": "\u1706" + VIRAMA,
    "ta": "\u1706",
    "te_ti": "\u1706" + KUDLIT_I,
    "to_tu": "\u1706" + KUDLIT_O,  # Keep for completeness even if absent in archive
}


def collect_images(root: Path) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = defaultdict(list)
    for path in sorted(root.glob("*.jpg")):
        prefix = path.name.split(".", 1)[0]
        groups[prefix].append(path)
    return groups


def tight_bbox(gray: Image.Image, threshold: int) -> Tuple[int, int, int, int]:
    arr = np.array(gray)
    mask = arr < threshold
    height, width = arr.shape

    if not mask.any():
        return 0, 0, width, height

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    top = int(rows[0])
    bottom = int(rows[-1] + 1)
    left = int(cols[0])
    right = int(cols[-1] + 1)
    # Clamp to bounds
    top = max(0, min(top, height - 1))
    bottom = max(top + 1, min(bottom, height))
    left = max(0, min(left, width - 1))
    right = max(left + 1, min(right, width))
    return left, top, right, bottom


def write_box(path: Path, label: str, bbox: Tuple[int, int, int, int], height: int) -> None:
    left, top, right, bottom = bbox
    bottom_inv = height - bottom
    top_inv = height - top
    line = f"{label} {left} {bottom_inv} {right} {top_inv} 0\n"
    path.write_text(line, encoding="utf-8")


def process_image(src: Path, dst_base: Path, label: str, threshold: int) -> None:
    with Image.open(src) as img:
        gray = img.convert("L")
    width, height = gray.size
    bbox = tight_bbox(gray, threshold)

    # Save grayscale TIFF
    gray.save(dst_base.with_suffix(".tif"), compression="tiff_deflate")

    # Write box
    write_box(dst_base.with_suffix(".box"), label, bbox, height)

    # Also save GT text for completeness
    dst_base.with_suffix(".gt.txt").write_text(label + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebox handwritten archive samples with kudlit/virama-aware labels.")
    parser.add_argument("--input", type=Path, required=True, help="Directory containing source JPEGs.")
    parser.add_argument("--output", type=Path, required=True, help="Destination directory for TIFF/BOX/GT triplets.")
    parser.add_argument("--threshold", type=int, default=230, help="Binary threshold for ink detection (0-255).")
    args = parser.parse_args()

    input_dir = args.input.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = collect_images(input_dir)
    total = sum(len(lst) for lst in groups.values())
    processed = 0
    skipped: List[Tuple[str, int]] = []

    for prefix in sorted(groups):
        label = LABEL_MAP.get(prefix)
        if label is None:
            skipped.append((prefix, len(groups[prefix])))
            continue
        for idx, src in enumerate(sorted(groups[prefix], key=lambda p: p.name)):
            dst_name = f"{prefix}_{idx:04d}"
            dst_base = output_dir / dst_name
            process_image(src, dst_base, label, args.threshold)
            processed += 1

    print(f"Processed {processed}/{total} samples into {output_dir}.")
    if skipped:
        print("Skipped prefixes without label mapping:")
        for prefix, count in skipped:
            print(f"  {prefix}: {count} files")


if __name__ == "__main__":
    main()
