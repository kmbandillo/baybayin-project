import argparse
import glob
import os
from typing import Optional, Tuple

import numpy as np
from PIL import Image


def otsu_threshold(arr: np.ndarray) -> int:
    """Compute an adaptive threshold using Otsu's method."""
    hist = np.bincount(arr.flatten(), minlength=256)
    total = arr.size
    sum_total = np.dot(np.arange(256), hist).astype(float)

    sum_background = 0.0
    weight_background = 0
    max_variance = 0.0
    threshold = 128

    for intensity in range(256):
        weight_background += hist[intensity]
        if weight_background == 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break

        sum_background += intensity * hist[intensity]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground

        variance_between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if variance_between > max_variance:
            max_variance = variance_between
            threshold = intensity

    return threshold


def find_indices(mask: np.ndarray, axis: int, min_pixels: int) -> Optional[Tuple[int, int]]:
    """
    Return bounding indices that keep the full glyph inside the box.

    We start from the widest range that contains any foreground pixels so even
    the thinnest extensions are preserved, then tighten inward based on
    min_pixels while expanding again whenever adjacent slim strokes exist.
    """
    counts = mask.sum(axis=axis)
    nonzero = np.where(counts > 0)[0]
    if nonzero.size == 0:
        return None

    if min_pixels <= 1:
        return int(nonzero[0]), int(nonzero[-1])

    robust = np.where(counts >= min_pixels)[0]
    if robust.size == 0:
        return int(nonzero[0]), int(nonzero[-1])

    first = int(robust[0])
    last = int(robust[-1])

    while first > 0 and counts[first - 1] > 0:
        first -= 1
    while last + 1 < counts.size and counts[last + 1] > 0:
        last += 1

    return first, last


def find_foreground_bbox(image: Image.Image, threshold: Optional[int] = None, min_pixels: int = 1) -> Optional[Tuple[int, int, int, int]]:
    """Return the (min_row, max_row, min_col, max_col) of pixels darker than threshold."""
    gray = image.convert("L")
    arr = np.asarray(gray)

    thresh = threshold if threshold is not None else otsu_threshold(arr)
    mask = arr <= thresh
    if not mask.any():
        return None

    row_span = find_indices(mask, axis=1, min_pixels=min_pixels)
    col_span = find_indices(mask, axis=0, min_pixels=min_pixels)
    if row_span is None or col_span is None:
        return None

    min_row, max_row = row_span
    min_col, max_col = col_span
    return min_row, max_row, min_col, max_col


def to_tesseract_box(min_row: int, max_row: int, min_col: int, max_col: int, width: int, height: int) -> Tuple[int, int, int, int]:
    """
    Convert Pillow-style coordinates to Tesseract box coordinates.

    Pillow uses top-left origin; Tesseract uses bottom-left origin with (left, bottom, right, top).
    """
    left = min_col
    right = min(max_col + 1, width)

    pillow_top = min_row
    pillow_bottom = min(max_row + 1, height)

    top = height - pillow_top
    bottom = height - pillow_bottom
    return left, bottom, right, top


def adjust_box_for_image(tif_path: str, box_path: str, threshold: Optional[int], min_pixels: int) -> bool:
    """Adjust the first box entry to tightly fit the glyph in tif_path."""
    with Image.open(tif_path) as img:
        width, height = img.size
        bbox = find_foreground_bbox(img, threshold=threshold, min_pixels=min_pixels)
        if bbox is None:
            print(f"warning: no foreground detected in {tif_path}")
            return False

    min_row, max_row, min_col, max_col = bbox
    left, bottom, right, top = to_tesseract_box(min_row, max_row, min_col, max_col, width, height)

    if not os.path.exists(box_path):
        print(f"warning: box file missing for {tif_path}")
        return False

    with open(box_path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]

    if not lines:
        print(f"warning: empty box file {box_path}")
        return False

    parts = lines[0].split(" ")
    if len(parts) < 6:
        print(f"warning: unexpected format in {box_path}")
        return False

    parts[1:5] = map(str, (left, bottom, right, top))
    lines[0] = " ".join(parts)

    with open(box_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")

    print(f"updated {os.path.basename(box_path)} -> ({left}, {bottom}, {right}, {top})")
    return True


def main():
    parser = argparse.ArgumentParser(description="Adjust Tesseract .box coordinates to tightly fit glyph pixels.")
    parser.add_argument("--dataset", "-d", required=True, help="Directory containing TIFF/.box pairs.")
    parser.add_argument("--prefix", "-p", default="a_", help="Filename prefix to match (default: a_).")
    parser.add_argument("--threshold", "-t", type=int, default=None, help="Optional grayscale threshold (0-255). Auto-detected with Otsu if omitted.")
    parser.add_argument("--min-pixels", "-m", type=int, default=3, help="Minimum dark pixels required per row/column when finding bounds.")
    args = parser.parse_args()

    pattern = os.path.join(args.dataset, f"{args.prefix}*.tif")
    tif_files = sorted(glob.glob(pattern))
    if not tif_files:
        print(f"No TIFF files found for pattern: {pattern}")
        return

    for tif_path in tif_files:
        box_path = os.path.splitext(tif_path)[0] + ".box"
        adjust_box_for_image(tif_path, box_path, threshold=args.threshold, min_pixels=args.min_pixels)


if __name__ == "__main__":
    main()
