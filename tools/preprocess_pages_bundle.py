#!/usr/bin/env python3
"""
Preprocess a multi-page TIFF bundle in-place without changing page dimensions.

The pipeline mirrors the handwritten preprocessing used elsewhere:
  * convert to grayscale
  * Gaussian blur (radius=1) for denoising
  * Otsu binarization
  * remove very small connected components

All pages remain the same size, and the output is saved as another multi-page TIFF.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import List

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
    arr = np.array(blurred, dtype=np.uint8)
    threshold = calculate_otsu_threshold(arr)
    binary = np.where(arr > threshold, 255, 0).astype(np.uint8)
    cleaned = remove_small_components(binary)
    return Image.fromarray(cleaned, mode="L")


def preprocess_bundle(input_path: Path, output_path: Path) -> None:
    with Image.open(input_path) as src:
        frames = [preprocess_frame(frame.copy()) for frame in ImageSequence.Iterator(src)]

    if not frames:
        raise ValueError(f"No frames found in {input_path}")

    first, rest = frames[0], frames[1:]
    first.save(
        output_path,
        save_all=True,
        append_images=rest,
        compression="tiff_deflate",
    )
    first.close()
    for frame in rest:
        frame.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess a multi-page TIFF without changing page sizes.")
    parser.add_argument("--input", type=Path, required=True, help="Source multi-page TIFF.")
    parser.add_argument("--output", type=Path, required=True, help="Destination multi-page TIFF.")
    args = parser.parse_args()

    preprocess_bundle(args.input.resolve(), args.output.resolve())
    print(f"Saved preprocessed bundle to {args.output}")


if __name__ == "__main__":
    main()
