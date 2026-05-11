#!/usr/bin/env python3
"""
Replace a range of pages in a multi-page TIFF with preprocessed versions from another bundle
while preserving (copying) the existing box annotations for those pages.

This is designed for Baybayin handwritten word bundles where early pages have been
manually corrected in `hw_words_pages.tif/.box` and need to be transferred into the
preprocessed bundle without altering bounding box coordinates.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from PIL import Image, ImageFilter, ImageSequence


def calculate_otsu_threshold(arr: np.ndarray) -> int:
    histogram, _ = np.histogram(arr, bins=256, range=(0, 256))
    total = arr.size
    sum_total = np.dot(np.arange(256), histogram)

    sum_background = 0.0
    weight_background = 0
    max_variance = 0.0
    threshold = 0

    for t in range(256):
        weight_background += histogram[t]
        if weight_background == 0:
            continue

        weight_foreground = total - weight_background
        if weight_foreground == 0:
            break

        sum_background += t * histogram[t]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

        if variance > max_variance:
            max_variance = variance
            threshold = t

    return threshold


def remove_small_components(arr: np.ndarray, fraction: float = 0.0001) -> np.ndarray:
    height, width = arr.shape
    min_component_size = max(1, int(height * width * fraction))

    visited = np.zeros((height, width), dtype=bool)
    black = arr == 0
    neighbors = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for y in range(height):
        for x in range(width):
            if not black[y, x] or visited[y, x]:
                continue
            queue: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            component: List[tuple[int, int]] = []

            while queue:
                cy, cx = queue.pop()
                component.append((cy, cx))
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width and black[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        queue.append((ny, nx))

            if len(component) < min_component_size:
                for cy, cx in component:
                    arr[cy, cx] = 255

    return arr


def preprocess_frame(frame: Image.Image) -> Image.Image:
    gray = frame.convert("L")
    blurred = gray.filter(ImageFilter.GaussianBlur(radius=1))
    arr = np.array(blurred)
    threshold = calculate_otsu_threshold(arr)
    binary = np.where(arr > threshold, 255, 0).astype(np.uint8)
    cleaned = remove_small_components(binary)
    return Image.fromarray(cleaned, mode="L")


def iter_frames(image: Image.Image) -> Iterable[Image.Image]:
    for frame in ImageSequence.Iterator(image):
        yield frame


def load_target_frames(path: Path) -> List[Image.Image]:
    with Image.open(path) as img:
        return [frame.copy().convert("L") for frame in iter_frames(img)]


def replace_pages(
    source_tif: Path,
    target_tif: Path,
    start_idx: int,
    end_idx: int,
) -> int:
    target_frames = load_target_frames(target_tif)
    total_pages = len(target_frames)
    if total_pages == 0:
        raise ValueError(f"No pages found in target TIFF: {target_tif}")
    if end_idx >= total_pages:
        raise ValueError(f"End index {end_idx} out of range for target with {total_pages} pages.")

    new_frames: List[Image.Image] = []

    with Image.open(source_tif) as src_img:
        for idx in range(total_pages):
            if start_idx <= idx <= end_idx:
                try:
                    src_img.seek(idx)
                except EOFError as exc:
                    raise ValueError(f"Source TIFF {source_tif} does not have page {idx}.") from exc
                frame = src_img.copy()
                processed = preprocess_frame(frame)
                new_frames.append(processed)
                frame.close()
            else:
                new_frames.append(target_frames[idx])

    first = new_frames[0]
    append_frames = new_frames[1:]
    first.save(
        target_tif,
        save_all=True,
        append_images=append_frames,
        compression="tiff_deflate",
    )
    first.close()
    for img in append_frames:
        img.close()

    return total_pages


def group_box_lines(path: Path) -> Dict[int, List[str]]:
    mapping: Dict[int, List[str]] = {}
    if not path.exists():
        return mapping
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            page = int(parts[-1])
        except (ValueError, IndexError):
            continue
        mapping.setdefault(page, []).append(line)
    return mapping


def update_box_file(
    source_box: Path,
    target_box: Path,
    start_idx: int,
    end_idx: int,
    total_pages: int,
) -> None:
    source_map = group_box_lines(source_box)
    target_map = group_box_lines(target_box)

    new_lines: List[str] = []
    for page in range(total_pages):
        if start_idx <= page <= end_idx:
            lines = source_map.get(page)
            if lines is None:
                raise ValueError(f"No box entries found for source page {page} in {source_box}.")
            new_lines.extend(lines)
        else:
            lines = target_map.get(page, [])
            new_lines.extend(lines)

    target_box.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replace pages in a multi-page TIFF bundle with preprocessed source pages.")
    parser.add_argument("--source-tif", type=Path, required=True, help="Source bundle with corrected pages.")
    parser.add_argument("--source-box", type=Path, required=True, help="Source BOX file containing corrected annotations.")
    parser.add_argument("--target-tif", type=Path, required=True, help="Target bundle to be updated in-place.")
    parser.add_argument("--target-box", type=Path, required=True, help="BOX file corresponding to the target bundle.")
    parser.add_argument("--start-page", type=int, default=1, help="First page to replace (1-based, inclusive).")
    parser.add_argument("--end-page", type=int, default=180, help="Last page to replace (1-based, inclusive).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.start_page < 1 or args.end_page < args.start_page:
        raise ValueError("Invalid page range.")

    start_idx = args.start_page - 1
    end_idx = args.end_page - 1

    total_pages = replace_pages(
        source_tif=args.source_tif.resolve(),
        target_tif=args.target_tif.resolve(),
        start_idx=start_idx,
        end_idx=end_idx,
    )

    update_box_file(
        source_box=args.source_box.resolve(),
        target_box=args.target_box.resolve(),
        start_idx=start_idx,
        end_idx=end_idx,
        total_pages=total_pages,
    )

    print(
        f"Replaced pages {args.start_page}-{args.end_page} in {args.target_tif} "
        f"and updated boxes in {args.target_box}."
    )


if __name__ == "__main__":
    main()
